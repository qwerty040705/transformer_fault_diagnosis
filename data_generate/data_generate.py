import sys, os, io, time, contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial.transform import Rotation as R

# ──────────────────── Project imports ────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dynamics.forward_kinematics_class import ForwardKinematics
from dynamics.lasdra_class import LASDRA
from planning.closed_loop_inverse_kinematics import ClosedLoopInverseKinematics
try:
    from fault_injection_fast import inject_faults_fast as inject_faults
except Exception:
    from fault_injection import inject_faults
from parameters import get_parameters
from parameters_model import parameters_model


# ---------------- Utils ----------------
def _quiet_call(func, *args, **kwargs):
    with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return func(*args, **kwargs)

def _format_hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _progress_bar(done, total, prefix="Generating", start_time=None):
    bar_len = 100
    frac = done / total if total else 1.0
    bar = "#" * int(bar_len * frac) + "-" * (bar_len - int(bar_len * frac))
    msg = f"\r{prefix} [{bar}] {frac*100:5.1f}% ({done}/{total})"
    if start_time:
        elapsed = time.time() - start_time
        remain  = (total - done) * elapsed / max(done, 1e-9)
        msg += f" | elapsed {_format_hms(elapsed)} | ETA {_format_hms(remain)}"
    print(msg, end="", flush=True)
    if done >= total:
        print()

def _is_finite(x): return np.all(np.isfinite(x))

def _conform_q(q, dof):
    q = np.asarray(q).reshape(-1, 1)
    if q.shape[0] < dof:
        q = np.vstack([q, np.zeros((dof - q.shape[0], 1))])
    return q[:dof]


# ---------------- Math helpers ----------------
def solve_ik(ik_solver, T_target, q_init, dof_target=None):
    if hasattr(ik_solver, "solve"):
        sol = ik_solver.solve(T_target, q_init)
    else:
        raise AttributeError("ClosedLoopInverseKinematics has no method 'solve'")

    if isinstance(sol, dict):
        if "q" in sol:
            q = sol["q"]
        elif "q_des" in sol:
            q = sol["q_des"]
        else:
            raise ValueError("IK returned dict without 'q' key.")
    else:
        q = sol

    q = np.asarray(q).reshape(-1, 1)
    if not _is_finite(q):
        q = np.asarray(q_init).reshape(-1, 1)
    if dof_target is not None:
        q = _conform_q(q, dof_target)
    return q


def solve_lambda_damped(H, tau, mu=1e-4):
    dof = H.shape[0]
    A = H @ H.T + mu * np.eye(dof)
    y = np.linalg.solve(A, tau)    
    lam = H.T @ y                   
    return lam


def clip_step(vec, max_abs=None):
    if max_abs is None:
        return vec
    s = np.linalg.norm(vec, ord=np.inf)
    if s > max_abs:
        vec = vec * (max_abs / (s + 1e-12))
    return vec





def random_unit():
    v = np.random.randn(3)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def random_so3():
    return R.from_rotvec(random_unit() * np.random.rand() * 2 * np.pi)

def align_z_to(vec_z):
    z = vec_z / (np.linalg.norm(vec_z) + 1e-12)
    x_tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x_tmp)
    y /= np.linalg.norm(y) + 1e-12
    x = np.cross(y, z)
    Rm = np.stack([x, y, z], axis=1)  
    return R.from_matrix(Rm)

def sample_on_cone(axis, alpha_max):
    u = random_unit()
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    cos_max = np.cos(alpha_max)
    while True:
        if np.dot(u, axis) >= cos_max:
            return u
        u = random_unit()

# ------------------------------------------------------------
# 메인 Trajectory generator
# ------------------------------------------------------------
def generate_random_se3_series(
    fk_solver,
    q0,
    link_count,
    link_length,
    T=200,
    max_pos_step=0.002,
    max_rot_step=0.01,
    max_try=30,
):
    """
    반환: [T,4,4] 리스트 (desired SE3 궤적)
    - n=1  : r=ℓ 구면, EE z축 = p̂, roll 자유
    - n=2  : (0≤r≤2ℓ)
        * r≤ℓ      : R ∈ SO(3) (완전)
        * ℓ<r<2ℓ   : z축이 cone(α<=acos(r/2ℓ)) + roll
        * r=2ℓ     : z축=p̂, roll
    - n≥3  : (0≤r≤nℓ)
        * s=nℓ-r ≥2ℓ → R ∈ SO(3)
        * 0<s<2ℓ    → z축 cone(α<=acos((r²+ℓ²-(n-1)²ℓ²)/(2rℓ)))+roll
        * s=0        → z축=p̂, roll
    """
    ℓ = float(link_length)
    n = link_count

    # 시작 프레임
    T0 = fk_solver.compute_end_effector_frame(q0.reshape(-1))
    R_cur = R.from_matrix(T0[:3, :3])
    p_cur = T0[:3, 3].copy()
    T_series = [T0]

    r_max = n * ℓ

    for _ in range(1, T):
        for _try in range(max_try):
            # ------------------------------------------------ position ------------------------------------------------
            p_cand = p_cur + np.random.uniform(-max_pos_step, max_pos_step, size=3)

            r = np.linalg.norm(p_cand)
            if n == 1:
                p_cand = p_cand / (r + 1e-12) * ℓ
                r = ℓ
            else:
                if r > r_max:
                    p_cand *= (r_max / (r + 1e-12))
                    r = r_max

            # ------------------------------------------------ orientation ---------------------------------------------
            if n == 1:
                roll = np.random.uniform(-max_rot_step, max_rot_step)
                R_new = R_cur * R.from_rotvec(np.array([0, 0, roll]))  # local roll
                R_align = align_z_to(p_cand / ℓ)
                R_new = R_align * R.from_rotvec(np.array([0, 0, roll]))

            elif n == 2:
                if r <= ℓ + 1e-12: 
                    R_new = random_so3()
                elif r >= 2*ℓ - 1e-12:  
                    roll = np.random.uniform(-max_rot_step, max_rot_step)
                    R_new = align_z_to(p_cand / r) * R.from_rotvec(np.array([0, 0, roll]))
                else:  
                    alpha_max = np.arccos(np.clip(r / (2*ℓ), -1.0, 1.0))
                    z_dir = sample_on_cone(p_cand / r, alpha_max)
                    roll = np.random.uniform(-max_rot_step, max_rot_step)
                    R_new = align_z_to(z_dir) * R.from_rotvec(np.array([0, 0, roll]))

            else:  
                s = n*ℓ - r
                if s >= 2*ℓ - 1e-12:        
                    R_new = random_so3()
                elif s <= 1e-12:             
                    roll = np.random.uniform(-max_rot_step, max_rot_step)
                    R_new = align_z_to(p_cand / r) * R.from_rotvec(np.array([0, 0, roll]))
                else:                          
                    alpha_max = np.arccos(
                        np.clip((r**2 + ℓ**2 - (n-1)**2 * ℓ**2) / (2*r*ℓ), -1.0, 1.0)
                    )
                    z_dir = sample_on_cone(p_cand / r, alpha_max)
                    roll = np.random.uniform(-max_rot_step, max_rot_step)
                    R_new = align_z_to(z_dir) * R.from_rotvec(np.array([0, 0, roll]))

            # ------------------------------------------------ append if finite ----------------------------------------
            T_new = np.eye(4)
            T_new[:3, :3] = R_new.as_matrix()
            T_new[:3, 3] = p_cand

            if np.all(np.isfinite(T_new)):
                T_series.append(T_new)
                p_cur, R_cur = p_cand, R_new
                break
        else:  
            T_series.append(T_series[-1].copy())

    return T_series



# ---------------- One sample generator ----------------
def generate_one_sample(link_count, T=200,     
                        epsilon_scale=0.05, dt=0.01, seed=None):
    if seed is not None: np.random.seed(seed)

    # ----------- model params -----------
    base_param = get_parameters(link_count)
    base_param['ODAR'] = base_param['ODAR'][:link_count]
    screw_axes, inertias = [], []
    for odar in base_param['ODAR']:
        screw_axes.extend(odar.body_joint_screw_axes)
        inertias.extend(odar.joint_inertia_tensor)
    base_param['LASDRA'].update(
        body_joint_screw_axes=screw_axes,
        inertia_matrix=inertias,
        dof=len(screw_axes)
    )
    model_param = _quiet_call(parameters_model, mode=0, params_prev=base_param)

    robot = LASDRA(model_param)
    fk_solver = ForwardKinematics(model_param)
    ik_solver = ClosedLoopInverseKinematics(model_param)
    dof       = model_param['LASDRA']['dof']
    link_len  = float(model_param['ODAR'][0].length)

    # ----------- desired SE(3) traj -----
    q0 = (np.random.rand(dof, 1) * 2 - 1) * 0.5 * np.pi
    T_des_series = generate_random_se3_series(
        fk_solver, q0, link_count, link_len, T=T,
        max_pos_step=0.001, max_rot_step=0.001
    )

    # ----------- IK forward propagate ---
    q_des  = np.zeros((T, dof, 1))
    dq_des = np.zeros_like(q_des)
    ddq_des= np.zeros_like(q_des)
    q_des[0] = solve_ik(ik_solver, T_des_series[0], q0, dof)
    for t in range(1, T):
        q_sol = solve_ik(ik_solver, T_des_series[t], q_des[t-1], dof)
        dq_tmp = clip_step((q_sol - q_des[t-1]) / dt, 1.0)
        q_des[t] = q_des[t-1] + dt * dq_tmp
        dq_des[t] = dq_tmp
        ddq_des[t] = (dq_des[t] - dq_des[t-1]) / dt
    dq_des  = np.clip(dq_des,  -5.0,  5.0)
    ddq_des = np.clip(ddq_des, -20.0, 20.0)

    # ----------- desired thrust ---------
    D_use = robot.D[:6*link_count]; B_use = robot.B_blkdiag[:6*link_count,:8*link_count]
    H = D_use.T @ B_use
    lam_des = np.zeros((T, 8*link_count))
    for t in range(T):
        robot.set_joint_states(q_des[t], dq_des[t])
        tau = robot.Mass @ ddq_des[t] + robot.Cori @ dq_des[t] + robot.Grav
        lam_des[t] = np.clip(
            solve_lambda_damped(H, tau).ravel(),
            -1.5*getattr(robot,"max_thrust",100.0),
             1.5*getattr(robot,"max_thrust",100.0)
        )

    # ----------- fault injection --------
    lam_faulty, type_matrix = inject_faults(
        lam_des, epsilon_scale=epsilon_scale
    )

    # ----------- forward dynamics --------
    actual_q  = np.zeros_like(q_des); actual_dq = np.zeros_like(dq_des)
    actual_q[0]=q_des[0]; robot.set_joint_states(actual_q[0], np.zeros_like(actual_q[0]))
    fk_actual = [fk_solver.compute_end_effector_frame(actual_q[0,:,0])]
    for t in range(1, T):
        thrust  = lam_faulty[t].reshape(-1,1)
        tau_flt = D_use.T @ (B_use @ thrust)
        q, dq = actual_q[t-1], actual_dq[t-1]
        robot.set_joint_states(q, dq)
        nxt = robot.get_next_joint_states(dt, tau_flt)
        qn, dqn = nxt['q'], np.clip(nxt['dq'], -10.0, 10.0)
        if not (_is_finite(qn) and _is_finite(dqn)):
            qn, dqn = q, dq
        actual_q[t], actual_dq[t] = qn, dqn
        robot.set_joint_states(qn, dqn)
        fk_actual.append(fk_solver.compute_end_effector_frame(qn[:,0]))

    desired_np = np.stack(T_des_series); actual_np = np.stack(fk_actual)
    label = np.tile(type_matrix.reshape(1,-1), (T,1))
    return desired_np, actual_np, label


def _worker(args):
    link_count, T, epsilon_scale, dt, seed = args
    return generate_one_sample(
        link_count=link_count, T=T,
        epsilon_scale=epsilon_scale, dt=dt, seed=seed
    )


def generate_dataset_parallel(link_count, T, NUM_SAMPLES, dt, epsilon_scale, workers=None):
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1

    args_list = [(link_count, T, epsilon_scale, dt, 1000+i) for i in range(NUM_SAMPLES)]

    desired_list, actual_list, label_list = [], [], []

    print(f"Spawning {workers} worker(s)...", flush=True)

    done_cnt = 0
    start_time = time.time()
    _progress_bar(done_cnt, NUM_SAMPLES, prefix="Generating samples", start_time=start_time)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, a) for a in args_list]
        for fut in as_completed(futs):
            res = fut.result()
            if isinstance(res, Exception):
                print("\n[Worker Error]", repr(res), flush=True)
                raise res
            d, a, l = res
            desired_list.append(d); actual_list.append(a); label_list.append(l)

            done_cnt += 1
            _progress_bar(done_cnt, NUM_SAMPLES, prefix="Generating samples", start_time=start_time)

    desired = np.asarray(desired_list)
    actual  = np.asarray(actual_list)
    label   = np.asarray(label_list)
    return desired, actual, label


# ---------------- Main ----------------
if __name__ == '__main__':
    link_count = int(input("How many links?: ").strip())

    try:
        T = int(input("Sequence length T?: ").strip()) 
    except Exception:
        T = 200

    try:
        NUM_SAMPLES = int(input("How many samples?: ").strip())
    except Exception:
        NUM_SAMPLES = 100

    
    # workers 입력: 0=자동(코어 수), 1=순차(디버깅), N=병렬 N개
    try:
        workers = int(input("How many workers? (0 = AUTO, WARNING: uses all CPU cores & more RAM): ").strip())
    except Exception:
        workers = 0
    if workers == 0:
        workers = os.cpu_count() or 1

    save_dir = os.path.join("data_storage", f"link_{link_count}")
    os.makedirs(save_dir, exist_ok=True)

    dt = 0.01
    epsilon_scale = 0.05

    timestamps = np.arange(T) * dt

    try:
        desired, actual, label = generate_dataset_parallel(
        link_count, T, NUM_SAMPLES, dt, epsilon_scale, workers
    )

        save_path = os.path.join(save_dir, "fault_dataset.npz")
        np.savez(save_path,
                 desired=desired,
                 actual=actual,
                 label=label,
                 timestamps=timestamps)
        print(f"Dataset saved successfully to {save_path}")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
        if 'desired' in locals() and len(desired) > 0:
            save_path = os.path.join(save_dir, "fault_dataset_partial.npz")
            np.savez(save_path,
                     desired=np.asarray(desired),
                     actual=np.asarray(actual),
                     label=np.asarray(label),
                     timestamps=timestamps)
            print(f"Partial dataset saved to {save_path}")
        else:
            print("No samples were generated. Nothing to save.")

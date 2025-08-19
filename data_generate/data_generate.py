import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp
import sys, io, time, contextlib
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

from control.impedance_controller import ImpedanceControllerMaximal
from control.external_actuation import ExternalActuation


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
    y = np.cross(z, x_tmp); y /= np.linalg.norm(y) + 1e-12
    x = np.cross(y, z)
    Rm = np.stack([x, y, z], axis=1)
    return R.from_matrix(Rm)
# === NEW: x축을 vec_x 방향에 정렬 ===
def align_x_to(vec_x):
    x = vec_x / (np.linalg.norm(vec_x) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    z0 = np.cross(x, ref); z0 /= (np.linalg.norm(z0) + 1e-12)
    y0 = np.cross(z0, x)
    Rm = np.stack([x, y0, z0], axis=1)  # 열이 [x, y, z]
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
# Desired SE(3) 궤적 생성기
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
            # ---------------- position ----------------
            p_cand = p_cur + np.random.uniform(-max_pos_step, max_pos_step, size=3)

            r = np.linalg.norm(p_cand)
            if n == 1:
                p_cand = p_cand / (r + 1e-12) * ℓ
                r = ℓ
            else:
                if r > r_max:
                    p_cand *= (r_max / (r + 1e-12))
                    r = r_max

            # ---------------- orientation --------------
            if n == 1:
                roll = np.random.uniform(-max_rot_step, max_rot_step)
                R_new = align_z_to(p_cand / ℓ) * R.from_rotvec(np.array([0, 0, roll]))

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

            # ---------------- append if finite ----------
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
def generate_one_sample(link_count, T=1000, epsilon_scale=0.0, dt=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # ---------------- 모델 파라미터/객체 ----------------
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

    # ---------------- 원하는 SE3 궤적 & IK ----------------
    q0 = (np.random.rand(dof, 1) * 2 - 1) * 0.5 * np.pi
    T_des_series = generate_random_se3_series(
        fk_solver, q0, link_count, link_len, T=T,
        max_pos_step=0.001, max_rot_step=0.001
    )

    q_des  = np.zeros((T, dof, 1))
    dq_des = np.zeros_like(q_des)
    ddq_des= np.zeros_like(q_des)
    q_des[0] = solve_ik(ik_solver, T_des_series[0], q0, dof)
    for t in range(1, T):
        q_sol = solve_ik(ik_solver, T_des_series[t], q_des[t-1], dof)
        dq_tmp = clip_step((q_sol - q_des[t-1]) / dt, 1.0)
        q_des[t]   = q_des[t-1] + dt * dq_tmp
        dq_des[t]  = dq_tmp
        ddq_des[t] = (dq_des[t] - dq_des[t-1]) / dt
    dq_des  = np.clip(dq_des,  -100.0,  100.0)
    ddq_des = np.clip(ddq_des, -100.0, 100.0)

    # ---------------- 컨트롤러/분배기 ----------------
    impedance_controller = ImpedanceControllerMaximal(model_param['ODAR'][-1])
    external_controller = ExternalActuation(model_param, robot)
    external_controller.apply_selective_mapping = False  # 우선 False로 검증

    # ---------------- 고장 계획(라벨만 미리 뽑기) ----------------
    # 주의: 여기서는 라벨 스케줄만 필요. lam_faulty는 쓰지 않는다.
    lam_dummy = np.zeros((T, 8*link_count))
    _, type_matrix, label_matrix, t0, idx, which_mask, onset_idx = inject_faults(
        lam_dummy, epsilon_scale=epsilon_scale, return_labels=True
    )

    # ---------------- 단 한 번의 폐루프 시뮬레이션 ----------------
    lam_log = np.zeros((T, 8*link_count))  # 실제 적용된 추력 로그(고장 반영)
    actual_q  = np.zeros((T, dof, 1))
    actual_dq = np.zeros_like(actual_q)
    actual_q[0] = q_des[0]
    robot.set_joint_states(actual_q[0], np.zeros_like(actual_q[0]))
    fk_actual = [fk_solver.compute_end_effector_frame(actual_q[0][:, 0])]

    impedance_controller.set_desired_pose(T_des_series[0])

    for t in range(1, T):
        # (1) 현재 실제 상태 기준
        q = actual_q[t-1]; dq = actual_dq[t-1]
        robot.set_joint_states(q, dq)
        T_act = fk_solver.compute_end_effector_frame(q[:, 0])
        R_EF  = T_act[:3, :3]

        # (2) Joint-space PD + Gravity compensation
        q_ref  = q_des[t]          # (dof,1)
        dq_ref = dq_des[t]         # (dof,1)

        # 게인은 상황 맞춰 조절 (너무 크면 진동, 너무 작으면 오차)
        Mjj = np.diag(robot.Mass)  # (dof,)
        fn, zeta = 2.5, 0.9        # 자연주파수 약 2.5 Hz, 감쇠 0.9
        wn = 2*np.pi*fn
        Kp_vec = (wn**2) * Mjj     # ≈ 246 * Mjj
        Kd_vec = (2*zeta*wn) * Mjj # ≈ 28 * Mjj

        # 루프 안
        e  = (q_ref - q).reshape(-1)
        de = (dq_ref - dq).reshape(-1)
        tau = (Kp_vec*e + Kd_vec*de) + robot.Grav.reshape(-1)

        # 안전하게 토크 클립 (분배기의 가용 추력 한계 고려)
        tau = np.clip(tau, -200000.0, 200000.0)


        # (6) 추력 분배 (LP→LS 폴백)
        lam_cmd = external_controller.distribute_torque_lp(tau)

        # === 노이즈 스케일 설정 ===
        # epsilon_scale: CLI 입력(예: 0.01 → 1% 표준편차)
        eps_norm  = float(epsilon_scale)        # 정상 모터 노이즈 비율
        eps_fault = float(epsilon_scale)        # 고장 모터 노이즈 비율(원하면 따로 크게)

        # === 모든 모터에 기본 노이즈 추가 ===
        ref = np.abs(lam_cmd) + 1e-9
        noise_all = np.random.normal(0.0, eps_norm, size=lam_cmd.shape) * ref

        lam_apply = lam_cmd + noise_all  # 항상 노이즈 적용

        # === 고장 모터: scale*lambda + noise 로 덮어쓰기 ===
        faulty = (label_matrix[t] == 0)
        if np.any(faulty):
            # scale=0 → stuck-off(+노이즈만), scale=1 → 정상(+노이즈)
            scale_fault = 0
            noise_fault = np.random.normal(0.0, eps_fault, size=lam_cmd.shape) * ref
            lam_apply[faulty] = scale_fault * lam_cmd[faulty] + noise_fault[faulty]

        lam_log[t] = lam_apply


        # (8) 실제 로봇에 적용 → 다음 상태 적분
        for i in range(link_count):
            thrust_i = lam_apply[8*i:8*(i+1)].reshape(-1, 1)
            robot.set_odar_body_wrench_from_thrust(thrust_i, i)
        tau_odar = robot.get_joint_torque_from_odars()

        nxt = robot.get_next_joint_states(dt, tau_odar)
        qn, dqn = nxt['q'], np.clip(nxt['dq'], -10.0, 10.0)
        if not (_is_finite(qn) and _is_finite(dqn)):
            qn, dqn = q, dq

        actual_q[t], actual_dq[t] = qn, dqn
        robot.set_joint_states(qn, dqn)
        fk_actual.append(fk_solver.compute_end_effector_frame(qn[:, 0]))

    # ---------------- 결과 패킹 ----------------
    # "desired"는 IK로 평활화한 q_des의 FK(학습 목표 포즈)로 두는 게 일관됨
    desired_np = np.stack([fk_solver.compute_end_effector_frame(q_des[t, :, 0]) for t in range(T)])
    actual_np  = np.stack(fk_actual)
    label      = label_matrix  # (T, 8*link_count)

    return desired_np, actual_np, label, which_mask, onset_idx, int(t0)



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
    which_mask_list, onset_idx_list, t0_list = [], [], []

    print(f"Spawning {workers} worker(s)...", flush=True)
    done_cnt = 0
    start_time = time.time()
    _progress_bar(done_cnt, NUM_SAMPLES, prefix="Generating samples", start_time=start_time)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        try:
            futs = [ex.submit(_worker, a) for a in args_list]
            for fut in as_completed(futs):
                res = fut.result()
                if isinstance(res, Exception):
                    print("\n[Worker Error]", repr(res), flush=True)
                    raise res

                d, a, l, which_mask, onset_idx, t0 = res
                desired_list.append(d)
                actual_list.append(a)
                label_list.append(l)
                which_mask_list.append(which_mask)
                onset_idx_list.append(onset_idx)
                t0_list.append(t0)

                done_cnt += 1
                _progress_bar(done_cnt, NUM_SAMPLES, prefix="Generating samples", start_time=start_time)

        except KeyboardInterrupt:
            # 남은 작업 취소하고 지금까지의 결과를 partial로 반환
            ex.shutdown(cancel_futures=True)
            print("\nInterrupted. Returning partial results...")
            if len(desired_list) == 0:
                # 아무 것도 쌓이지 않았다면 빈 결과를 partial로 리턴
                return (
                    np.empty((0,)), np.empty((0,)), np.empty((0,)),
                    np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=np.int32),
                    True  # is_partial
                )

            desired = np.asarray(desired_list)               # (S_partial,T,4,4)
            actual  = np.asarray(actual_list)                # (S_partial,T,4,4)
            label   = np.asarray(label_list)                 # (S_partial,T,M)
            which_fault_mask = np.asarray(which_mask_list)   # (S_partial,M)
            onset_idx = np.asarray(onset_idx_list)           # (S_partial,M)
            t0_arr    = np.asarray(t0_list, dtype=np.int32)  # (S_partial,)
            return desired, actual, label, which_fault_mask, onset_idx, t0_arr, True  # partial

    # 정상 완료 시
    desired = np.asarray(desired_list)
    actual  = np.asarray(actual_list)
    label   = np.asarray(label_list)
    which_fault_mask = np.asarray(which_mask_list)
    onset_idx = np.asarray(onset_idx_list)
    t0_arr    = np.asarray(t0_list, dtype=np.int32)
    return desired, actual, label, which_fault_mask, onset_idx, t0_arr, False  # not partial



# ---------------- Main ----------------
if __name__ == '__main__':
    link_count = int(input("How many links?: ").strip())

    try:
        T = int(input("Sequence length T? (default 1000): ").strip())
    except Exception:
        T = 1000

    try:
        NUM_SAMPLES = int(input("How many samples?: ").strip())
    except Exception:
        NUM_SAMPLES = 100

    try:
        workers = int(input("How many workers? (0 = AUTO, WARNING: uses all CPU cores & more RAM): ").strip())
    except Exception:
        workers = 0
    if workers == 0:
        workers = os.cpu_count() or 1

    try:
        epsilon_scale = float(input("Epsilon scale (0 for none) [default 0.0]: ").strip())
    except Exception:
        epsilon_scale = 0.0

    save_dir = os.path.join("data_storage", f"link_{link_count}")
    os.makedirs(save_dir, exist_ok=True)

    dt = 0.01
    timestamps = np.arange(T) * dt

    desired, actual, label, which_mask, onset_idx, t0_arr, is_partial = generate_dataset_parallel(
        link_count, T, NUM_SAMPLES, dt, epsilon_scale, workers
    )

    if desired.size == 0:
        print("No samples were generated. Nothing to save.")
    else:
        fname = "fault_dataset_partial.npz" if is_partial else "fault_dataset.npz"
        save_path = os.path.join(save_dir, fname)
        np.savez(
            save_path,
            desired=desired,            # (S,T,4,4)
            actual=actual,              # (S,T,4,4)
            label=label,                # (S,T,M)
            which_fault_mask=which_mask,# (S,M)
            onset_idx=onset_idx,        # (S,M)
            t0=t0_arr,                  # (S,)
            timestamps=timestamps,      # (T,)
            dt=dt,
            link_count=link_count
        )
        print(f"\nDataset saved successfully to {save_path}")

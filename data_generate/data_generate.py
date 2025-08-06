# data_generate/data_generate.py
import sys
import os
import io
import time
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- repo import path (자식 프로세스에서도 유효) ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dynamics.forward_kinematics_class import ForwardKinematics
from dynamics.lasdra_class import LASDRA
from planning.closed_loop_inverse_kinematics import ClosedLoopInverseKinematics
# 빠른 fault 주입이 있으면 사용, 없으면 기존 함수 사용
try:
    from fault_injection_fast import inject_faults_fast as inject_faults
except Exception:
    from fault_injection import inject_faults

from parameters import get_parameters
from parameters_model import parameters_model
from scipy.spatial.transform import Rotation as R


# ---------------- Utils ----------------
def _quiet_call(func, *args, **kwargs):
    with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return func(*args, **kwargs)

def _format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _progress_bar(done, total, prefix="Generating samples", start_time=None):
    """
    단일 진행바(100칸) + 경과/ETA. 같은 줄에서 갱신.
    """
    bar_len = 100
    done = max(0, min(done, total))
    frac = (done / total) if total > 0 else 1.0
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    msg = f"\r{prefix} [{bar}] {frac*100:5.1f}% ({done}/{total})"
    if start_time is not None:
        elapsed = time.time() - start_time
        msg += f" | elapsed {_format_hms(elapsed)}"
        if done > 0:
            rate = done / max(elapsed, 1e-9)
            remain = (total - done) / rate
            msg += f" | ETA {_format_hms(remain)}"
    print(msg, end="", flush=True)
    if done >= total:
        print()

def _is_finite(x):
    return np.all(np.isfinite(x))

def _conform_q(q, dof):
    q = np.asarray(q).reshape(-1, 1)
    if q.shape[0] > dof:
        return q[:dof, :]
    if q.shape[0] < dof:
        pad = np.zeros((dof - q.shape[0], 1), dtype=float)
        return np.vstack([q, pad])
    return q


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
    # H: [dof x m], tau: [dof x 1]
    dof = H.shape[0]
    A = H @ H.T + mu * np.eye(dof)
    y = np.linalg.solve(A, tau)      # [dof x 1]
    lam = H.T @ y                    # [m x 1]
    return lam


def clip_step(vec, max_abs=None):
    if max_abs is None:
        return vec
    s = np.linalg.norm(vec, ord=np.inf)
    if s > max_abs:
        vec = vec * (max_abs / (s + 1e-12))
    return vec


# ---------------- Trajectory generator ----------------
def generate_random_se3_series(
    fk_solver,
    q0,
    link_count,
    link_length,
    T=200,
    max_pos_step=0.001,
    max_rot_step=0.001
):
    """
    - link_count == 1: 반지름 = link_length 구면(S^2) 위 이동 (orientation 고정)
    - link_count >= 2: 반지름 r_max = link_count * link_length 의 구(B^3) 내부 이동, 소규모 회전 허용
    """
    T0 = fk_solver.compute_end_effector_frame(q0.reshape(-1))
    R_cur = R.from_matrix(T0[:3, :3])
    p_cur = T0[:3, 3].copy()
    T_series = [T0]

    r_max = link_count * float(link_length)
    r_eps = 1e-6

    for _ in range(1, T):
        for _try in range(12):
            dp = np.random.uniform(-max_pos_step, max_pos_step, size=3)
            p_cand = p_cur + dp

            if link_count == 1:
                normp = np.linalg.norm(p_cand)
                if normp > r_eps:
                    p_cand = p_cand / normp * link_length
                else:
                    p_cand = np.array([link_length, 0, 0], dtype=float)
                R_new = R_cur
            else:
                normp = np.linalg.norm(p_cand)
                if normp > r_max:
                    p_cand = p_cand * (r_max / (normp + 1e-12))

                axis = np.random.randn(3)
                norm = np.linalg.norm(axis)
                axis = axis / norm if norm >= 1e-12 else np.array([1.0, 0.0, 0.0])
                angle = np.random.uniform(-max_rot_step, max_rot_step)
                R_step = R.from_rotvec(axis * angle)
                R_new = R_cur * R_step

            T_new = np.eye(4)
            T_new[:3, :3] = R_new.as_matrix()
            T_new[:3, 3] = p_cand

            if np.all(np.isfinite(T_new)):
                T_series.append(T_new)
                R_cur = R_new
                p_cur = p_cand
                break
        else:
            T_series.append(T_series[-1].copy())

    return T_series


# ---------------- One sample generator ----------------
def generate_one_sample(link_count, T=200, fault_time=100, epsilon_scale=0.05, dt=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)

    base_param = get_parameters(link_count)
    base_param['ODAR'] = base_param['ODAR'][:link_count]

    screw_axes_all, inertia_all = [], []
    for odar in base_param['ODAR']:
        screw_axes_all.extend(odar.body_joint_screw_axes)
        inertia_all.extend(odar.joint_inertia_tensor)
    base_param['LASDRA']['body_joint_screw_axes'] = screw_axes_all
    base_param['LASDRA']['inertia_matrix'] = inertia_all
    base_param['LASDRA']['dof'] = len(screw_axes_all)

    model_param = _quiet_call(parameters_model, mode=0, params_prev=base_param)

    robot = LASDRA(model_param)
    fk_solver = ForwardKinematics(model_param)
    ik_solver = ClosedLoopInverseKinematics(model_param)

    num_links = link_count
    dof = model_param['LASDRA']['dof']

    try:
        link_len = float(model_param['ODAR'][0].length)
    except Exception:
        link_len = float(model_param['ODAR'][0]['length'])

    # 목표 SE3 시퀀스 생성
    q0 = (2 * np.random.rand(dof, 1) - 1) * (0.5 * np.pi)
    T_des_series = generate_random_se3_series(
        fk_solver, q0, num_links, link_len, T=T, max_pos_step=0.001, max_rot_step=0.001
    )

    # IK로 q_des 생성 (semi-implicit)
    q_des  = np.zeros((T, dof, 1))
    dq_des = np.zeros((T, dof, 1))
    ddq_des = np.zeros((T, dof, 1))
    q_des[0] = solve_ik(ik_solver, T_des_series[0], q0, dof_target=dof)

    for t in range(1, T):
        q_init = q_des[t-1]
        q_sol = solve_ik(ik_solver, T_des_series[t], q_init, dof_target=dof)
        dq_tmp = (q_sol - q_des[t-1]) / dt
        dq_tmp = clip_step(dq_tmp, max_abs=1.0)
        q_des[t] = q_des[t-1] + dt * dq_tmp
        dq_des[t] = dq_tmp
        ddq_des[t] = (dq_des[t] - dq_des[t-1]) / dt

    dq_des  = np.clip(dq_des,  -5.0,  5.0)
    ddq_des = np.clip(ddq_des, -20.0, 20.0)

    # 원하는 추력(lambda_des) 계산
    lambda_des = np.zeros((T, 8 * num_links))
    mu_alloc = 1e-4

    # 고정 행렬 미리 준비
    D_use = robot.D[:6 * num_links, :]                       # [6N x dof]
    B_use = robot.B_blkdiag[:6 * num_links, :8 * num_links]  # [6N x 8N]
    H = D_use.T @ B_use                                      # [dof x 8N]

    for t in range(T):
        robot.set_joint_states(q_des[t], dq_des[t])
        tau = robot.Mass @ ddq_des[t] + robot.Cori @ dq_des[t] + robot.Grav
        lam_t = solve_lambda_damped(H, tau, mu=mu_alloc)
        lam_t = np.clip(
            lam_t,
            -getattr(robot, "max_thrust", 100.0) * 1.5,
             getattr(robot, "max_thrust", 100.0) * 1.5
        )
        lambda_des[t, :] = lam_t.reshape(-1)

    # 고장 주입
    lambda_faulty, type_matrix = inject_faults(
        lambda_des, fault_time=fault_time, epsilon_scale=epsilon_scale
    )

    # 실제 궤적 적분
    actual_q = np.zeros_like(q_des)
    actual_dq = np.zeros_like(dq_des)
    actual_q[0] = q_des[0]
    actual_dq[0] = np.zeros_like(q_des[0])

    robot.set_joint_states(actual_q[0], actual_dq[0])
    T_actual_series = [fk_solver.compute_end_effector_frame(actual_q[0, :, 0])]

    for t in range(1, T):
        thrust_vec = lambda_faulty[t, :].reshape(-1, 1)      # [8N x 1]
        F_total = B_use @ thrust_vec                         # [6N x 1]
        tau_fault = D_use.T @ F_total                        # [dof x 1]

        q_curr = actual_q[t-1]; dq_curr = actual_dq[t-1]
        robot.set_joint_states(q_curr, dq_curr)

        success = False
        for sub_div in [1, 2, 4, 8]:
            dt_local = dt / sub_div
            q_tmp = q_curr.copy(); dq_tmp = dq_curr.copy()
            robot.set_joint_states(q_tmp, dq_tmp)
            ok = True
            for _ in range(sub_div):
                nxt = robot.get_next_joint_states(dt_local, tau_fault)
                q_tmp = nxt['q']; dq_tmp = nxt['dq']
                if (not _is_finite(q_tmp)) or (not _is_finite(dq_tmp)):
                    ok = False; break
                dq_tmp = np.clip(dq_tmp, -10.0, 10.0)
                robot.set_joint_states(q_tmp, dq_tmp)
            if ok:
                actual_q[t] = q_tmp; actual_dq[t] = dq_tmp
                success = True; break

        if not success:
            actual_q[t] = actual_q[t-1]
            actual_dq[t] = actual_dq[t-1]

        robot.set_joint_states(actual_q[t], actual_dq[t])
        T_actual_series.append(fk_solver.compute_end_effector_frame(actual_q[t, :, 0]))

    # 라벨 생성
    label = np.ones((T, 8 * num_links), dtype=int)
    label_fault = type_matrix.reshape(-1)
    if fault_time < T:
        label[fault_time:, :] = np.tile(label_fault, (T - fault_time, 1))

    desired_np = np.stack(T_des_series, axis=0)
    actual_np  = np.stack(T_actual_series, axis=0)
    return desired_np, actual_np, label


# ---------------- 병렬 생성 래퍼 ----------------
def _worker(args):
    (link_count, T, fault_time, epsilon_scale, dt, seed) = args
    try:
        return generate_one_sample(
            link_count=link_count, T=T, fault_time=fault_time,
            epsilon_scale=epsilon_scale, dt=dt, seed=seed
        )
    except Exception as e:
        return e  # 예외를 상위로 넘겨 디버깅 쉽게


def generate_dataset_parallel(link_count, T, FAULT_TIME, NUM_SAMPLES, dt, epsilon_scale, workers=None):
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1

    args_list = [(link_count, T, FAULT_TIME, epsilon_scale, dt, 1000 + i)
                 for i in range(NUM_SAMPLES)]
    desired_list, actual_list, label_list = [], [], []

    # 안내 메시지 (개행 포함, 진행바와 겹치지 않도록)
    print(f"Spawning {workers} worker(s)...", flush=True)

    # 시작 직후 0% 진행바 한 번 표시
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
        NUM_SAMPLES = int(input("How many samples?: ").strip())
    except Exception:
        NUM_SAMPLES = 100  # fallback

    # workers 입력: 0=자동(코어 수), 1=순차(디버깅), N=병렬 N개
    try:
        workers = int(input("How many workers? (0 = AUTO, WARNING: uses all CPU cores & more RAM): ").strip())
    except Exception:
        workers = 0
    if workers == 0:
        workers = os.cpu_count() or 1

    save_dir = os.path.join("data_storage", f"link_{link_count}")
    os.makedirs(save_dir, exist_ok=True)

    # 기본 파라미터
    T = 200
    FAULT_TIME = 100
    dt = 0.01
    epsilon_scale = 0.05

    timestamps = np.arange(T) * dt

    try:
        desired, actual, label = generate_dataset_parallel(
            link_count=link_count,
            T=T,
            FAULT_TIME=FAULT_TIME,
            NUM_SAMPLES=NUM_SAMPLES,
            dt=dt,
            epsilon_scale=epsilon_scale,
            workers=workers
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

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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

def clip_step(vec, max_abs=None):
    if max_abs is None:
        return vec
    s = np.linalg.norm(vec, ord=np.inf)
    if s > max_abs:
        vec = vec * (max_abs / (s + 1e-12))
    return vec


# ---------------- Math helpers ----------------
def solve_ik(ik_solver, T_target, q_init, dof_target=None):
    if hasattr(ik_solver, "solve"):
        sol = ik_solver.solve(T_target, q_init)
    else:
        raise AttributeError("ClosedLoopInverseKinematics has no method 'solve'")
    if isinstance(sol, dict):
        if "q" in sol: q = sol["q"]
        elif "q_des" in sol: q = sol["q_des"]
        else: raise ValueError("IK returned dict without 'q' key.")
    else:
        q = sol
    q = np.asarray(q).reshape(-1, 1)
    if not _is_finite(q):
        q = np.asarray(q_init).reshape(-1, 1)
    if dof_target is not None:
        q = _conform_q(q, dof_target)
    return q

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

def sample_on_cone(axis, alpha_max):
    u = random_unit()
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    cos_max = np.cos(alpha_max)
    while True:
        if np.dot(u, axis) >= cos_max:
            return u
        u = random_unit()


# ------------------------------------------------------------
# Desired SE(3) 궤적 생성기  (그대로 유지)
# ------------------------------------------------------------
def generate_random_se3_series(
    fk_solver, q0, link_count, link_length, T=200,
    max_pos_step=0.002, max_rot_step=0.01, max_try=30,
):
    ℓ = float(link_length)
    n = link_count
    T0 = fk_solver.compute_end_effector_frame(q0.reshape(-1))
    R_cur = R.from_matrix(T0[:3, :3])
    p_cur = T0[:3, 3].copy()
    T_series = [T0]
    r_max = n * ℓ

    for _ in range(1, T):
        for _try in range(max_try):
            p_cand = p_cur + np.random.uniform(-max_pos_step, max_pos_step, size=3)
            r = np.linalg.norm(p_cand)
            if n == 1:
                p_cand = p_cand / (r + 1e-12) * ℓ
            else:
                if r > r_max:
                    p_cand *= (r_max / (r + 1e-12))
                    r = r_max
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


# ---------------- helpers: per-link transforms (순수 링크 상대/누적)
def _exp_se3_from_Aset(fk_solver, j_idx, theta):
    xi = fk_solver.Aset[:, j_idx]
    return ForwardKinematics._exp_se3(xi, float(theta))

def compute_link_relatives_pure(fk_solver, q_vec):
    q_flat = np.asarray(q_vec).reshape(-1)
    link_T = []
    joint_counts = [len(odar.body_joint_screw_axes) for odar in fk_solver.ODAR]
    j0 = 0
    for k, nj in enumerate(joint_counts):
        T_link = np.eye(4)
        for j in range(nj):
            T_link = T_link @ _exp_se3_from_Aset(fk_solver, j0 + j, q_flat[j0 + j])
        T_link = T_link @ fk_solver.T_joint_to_joint_set[k]
        link_T.append(T_link)
        j0 += nj
    return link_T, joint_counts

def compute_link_cumulative_from_rel(link_rel_list):
    cum = []
    T = np.eye(4)
    for Trel in link_rel_list:
        T = T @ Trel
        cum.append(T.copy())
    return cum


# ---------------- One sample generator ----------------
def generate_one_sample(link_count, T=1000, epsilon_scale=0.0, dt=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)

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

    # 원하는 SE3 궤적 & IK
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

    # 컨트롤 & 외력 분배
    impedance_controller = ImpedanceControllerMaximal(model_param['ODAR'][-1])
    external_controller = ExternalActuation(model_param, robot)
    external_controller.apply_selective_mapping = False

    # ── [CHANGE #3-1] 건강 구간에서 saturation을 피하기 위한 상한 스케일 ──
    THRUST_BOUND_SCALE = 50.0  # 필요시 10~200 사이 조절
    if hasattr(external_controller, "lp") and hasattr(external_controller, "qp"):
        if "ub" in external_controller.qp and "lb" in external_controller.qp:
            external_controller.qp["ub"] *= THRUST_BOUND_SCALE
            external_controller.qp["lb"] *= THRUST_BOUND_SCALE
        if "ub" in external_controller.lp and "lb" in external_controller.lp:
            # lp["ub"]는 [λ; t] 구조 — 마지막 t는 제외하고 스케일
            external_controller.lp["ub"][:-1] *= THRUST_BOUND_SCALE
            external_controller.lp["lb"][:-1] *= THRUST_BOUND_SCALE

    # 고장 시나리오
    lam_dummy = np.zeros((T, 8*link_count))
    _, type_matrix, label_matrix, t0, idx, which_mask, onset_idx = inject_faults(
        lam_dummy, epsilon_scale=epsilon_scale, return_labels=True
    )

    # 로그 버퍼
    actual_q  = np.zeros((T, dof, 1))
    actual_dq = np.zeros_like(actual_q)
    actual_q[0] = q_des[0]
    robot.set_joint_states(actual_q[0], np.zeros_like(actual_q[0]))

    # EE 버퍼
    desired_ee = np.zeros((T, 4, 4))
    actual_ee  = np.zeros((T, 4, 4))

    # ── [CHANGE #1] EE(desired)는 항상 FK(q_des)로 계산
    desired_ee[0] = fk_solver.compute_end_effector_frame(q_des[0][:, 0])
    actual_ee[0]  = fk_solver.compute_end_effector_frame(actual_q[0][:, 0])

    # 링크 상대/누적 버퍼
    desired_link_rel = np.zeros((T, link_count, 4, 4))
    actual_link_rel  = np.zeros((T, link_count, 4, 4))
    desired_link_cum = np.zeros((T, link_count, 4, 4))
    actual_link_cum  = np.zeros((T, link_count, 4, 4))

    # t=0 기록 (순수 링크상대 계산)
    link_rel_des0, joint_counts = compute_link_relatives_pure(fk_solver, q_des[0][:, 0])
    link_rel_act0, _            = compute_link_relatives_pure(fk_solver, actual_q[0][:, 0])
    link_cum_des0 = compute_link_cumulative_from_rel(link_rel_des0)
    link_cum_act0 = compute_link_cumulative_from_rel(link_rel_act0)
    for k in range(link_count):
        desired_link_rel[0, k] = link_rel_des0[k]
        actual_link_rel[0, k]  = link_rel_act0[k]
        desired_link_cum[0, k] = link_cum_des0[k]
        actual_link_cum[0, k]  = link_cum_act0[k]

    dt = 0.01

    for t in range(1, T):
        q = actual_q[t-1]; dq = actual_dq[t-1]
        robot.set_joint_states(q, dq)

        # ── [CHANGE #3-2] PD 게인 살짝 낮춤(요구 토크 완화)
        Mjj = np.diag(robot.Mass)
        fn, zeta = 0.8, 0.95   # ← 기존 2.5, 0.9 에서 완화
        wn = 2*np.pi*fn
        Kp_vec = (wn**2) * Mjj
        Kd_vec = (2*zeta*wn) * Mjj

        e  = (q_des[t]  - q ).reshape(-1)
        de = (dq_des[t] - dq).reshape(-1)
        tau = (Kp_vec*e + Kd_vec*de) + robot.Grav.reshape(-1)

        # 과도 토크 제한 — 필요시 조금 더 완화
        tau = np.clip(tau, -5e4, 5e4)

        # 스러스트 분배 + 노이즈/고장
        lam_cmd = external_controller.distribute_torque_lp(tau)

        eps = float(epsilon_scale)
        ref = np.abs(lam_cmd) + 1e-9
        noise_all = np.random.normal(0.0, eps, size=lam_cmd.shape) * ref
        lam_apply = lam_cmd + noise_all

        faulty = (label_matrix[t] == 0)
        if np.any(faulty):
            scale_fault = 0.0  # stuck-off
            noise_fault = np.random.normal(0.0, eps, size=lam_cmd.shape) * ref
            lam_apply[faulty] = scale_fault * lam_cmd[faulty] + noise_fault[faulty]

        for iL in range(link_count):
            thrust_i = lam_apply[8*iL:8*(iL+1)].reshape(-1, 1)
            robot.set_odar_body_wrench_from_thrust(thrust_i, iL)
        tau_odar = robot.get_joint_torque_from_odars()

        # 적분
        nxt = robot.get_next_joint_states(dt, tau_odar)
        qn, dqn = nxt['q'], np.clip(nxt['dq'], -10.0, 10.0)
        if not (_is_finite(qn) and _is_finite(dqn)):
            qn, dqn = q, dq

        actual_q[t], actual_dq[t] = qn, dqn
        robot.set_joint_states(qn, dqn)

        # EE 기록 (FK 기반)
        desired_ee[t] = fk_solver.compute_end_effector_frame(q_des[t, :, 0])
        actual_ee[t]  = fk_solver.compute_end_effector_frame(qn[:, 0])

        # ── [CHANGE #2] 링크 상대/누적 — Aset로 순수 계산
        link_rel_des, _ = compute_link_relatives_pure(fk_solver, q_des[t, :, 0])
        link_rel_act, _ = compute_link_relatives_pure(fk_solver, qn[:, 0])
        link_cum_des    = compute_link_cumulative_from_rel(link_rel_des)
        link_cum_act    = compute_link_cumulative_from_rel(link_rel_act)
        for k in range(link_count):
            desired_link_rel[t, k] = link_rel_des[k]
            actual_link_rel[t, k]  = link_rel_act[k]
            desired_link_cum[t, k] = link_cum_des[k]
            actual_link_cum[t, k]  = link_cum_act[k]

    return (
        desired_ee, actual_ee, label_matrix, which_mask, onset_idx, int(t0),
        desired_link_rel, actual_link_rel, desired_link_cum, actual_link_cum,
        dof, np.array(joint_counts, dtype=np.int32)
    )


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

    desired_ee_list, actual_ee_list, label_list = [], [], []
    which_mask_list, onset_idx_list, t0_list = [], [], []
    d_link_rel_list, a_link_rel_list = [], []
    d_link_cum_list, a_link_cum_list = [], []
    dof_list, joint_counts_list = [], []

    print(f"Spawning {workers} worker(s)...", flush=True)
    done_cnt = 0
    start_time = time.time()
    _progress_bar(done_cnt, NUM_SAMPLES, prefix="Generating samples", start_time=start_time)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        try:
            futs = [ex.submit(_worker, a) for a in args_list]
            for fut in as_completed(futs):
                (d_ee, a_ee, l, which_mask, onset_idx, t0,
                 d_lr, a_lr, d_lc, a_lc, dof, joint_counts) = fut.result()

                desired_ee_list.append(d_ee)
                actual_ee_list.append(a_ee)
                label_list.append(l)
                which_mask_list.append(which_mask)
                onset_idx_list.append(onset_idx)
                t0_list.append(t0)
                d_link_rel_list.append(d_lr)
                a_link_rel_list.append(a_lr)
                d_link_cum_list.append(d_lc)
                a_link_cum_list.append(a_lc)
                dof_list.append(dof)
                joint_counts_list.append(joint_counts)

                done_cnt += 1
                _progress_bar(done_cnt, NUM_SAMPLES, prefix="Generating samples", start_time=start_time)

        except KeyboardInterrupt:
            ex.shutdown(cancel_futures=True)
            print("\nInterrupted. Returning partial results...")
            if len(desired_ee_list) == 0:
                return (
                    np.empty((0,)), np.empty((0,)), np.empty((0,)),
                    np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=np.int32),
                    np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,)),
                    0, np.empty((0,), dtype=np.int32),
                    True
                )

            desired_ee = np.asarray(desired_ee_list)
            actual_ee  = np.asarray(actual_ee_list)
            label      = np.asarray(label_list)
            which_fault_mask = np.asarray(which_mask_list)
            onset_idx  = np.asarray(onset_idx_list)
            t0_arr     = np.asarray(t0_list, dtype=np.int32)
            d_link_rel = np.asarray(d_link_rel_list)
            a_link_rel = np.asarray(a_link_rel_list)
            d_link_cum = np.asarray(d_link_cum_list)
            a_link_cum = np.asarray(a_link_cum_list)
            dof_out    = int(dof_list[0]) if dof_list else 0
            joint_counts_arr = np.asarray(joint_counts_list)
            return (desired_ee, actual_ee, label, which_fault_mask, onset_idx, t0_arr,
                    d_link_rel, a_link_rel, d_link_cum, a_link_cum, dof_out, joint_counts_arr,
                    True)

    desired_ee = np.asarray(desired_ee_list)
    actual_ee  = np.asarray(actual_ee_list)
    label      = np.asarray(label_list)
    which_fault_mask = np.asarray(which_mask_list)
    onset_idx  = np.asarray(onset_idx_list)
    t0_arr     = np.asarray(t0_list, dtype=np.int32)
    d_link_rel = np.asarray(d_link_rel_list)
    a_link_rel = np.asarray(a_link_rel_list)
    d_link_cum = np.asarray(d_link_cum_list)
    a_link_cum = np.asarray(a_link_cum_list)
    dof_out    = int(dof_list[0]) if dof_list else 0
    joint_counts_arr = np.asarray(joint_counts_list)
    return (desired_ee, actual_ee, label, which_fault_mask, onset_idx, t0_arr,
            d_link_rel, a_link_rel, d_link_cum, a_link_cum, dof_out, joint_counts_arr,
            False)


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
        workers = int(input("How many workers? (0 = AUTO): ").strip())
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

    (desired_ee, actual_ee, label, which_mask, onset_idx, t0_arr,
     d_link_rel, a_link_rel, d_link_cum, a_link_cum, dof, joint_counts_arr,
     is_partial) = generate_dataset_parallel(
        link_count, T, NUM_SAMPLES, dt, epsilon_scale, workers
    )

    if desired_ee.size == 0:
        print("No samples were generated. Nothing to save.")
    else:
        fname = "fault_dataset_partial.npz" if is_partial else "fault_dataset.npz"
        save_path = os.path.join(save_dir, fname)
        np.savez(
            save_path,
            desired_link_rel=d_link_rel,
            actual_link_rel=a_link_rel,
            desired_link_cum=d_link_cum,
            actual_link_cum=a_link_cum,
            label=label,
            which_fault_mask=which_mask,
            onset_idx=onset_idx,
            t0=t0_arr,
            timestamps=timestamps,
            dt=dt,
            link_count=link_count,
            dof=dof,
            joint_counts=joint_counts_arr
        )
        print(f"\nDataset saved successfully to {save_path}")

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
import sys
import io
import time
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dynamics.forward_kinematics_class import ForwardKinematics
from dynamics.lasdra_class import LASDRA
from planning.closed_loop_inverse_kinematics import ClosedLoopInverseKinematics  # (더이상 사용X, 호환용)
try:
    from fault_injection_fast import inject_faults_fast as inject_faults
except Exception:
    from fault_injection import inject_faults
from parameters import get_parameters
from parameters_model import parameters_model

from control.impedance_controller import ImpedanceControllerMaximal
from control.external_actuation import ExternalActuation


FORCE_IDEAL_WHEN_HEALTHY = False
CHECK_DISTRIBUTION_RESID = True
DEBUG_TAU_PRINT = False

IK_MAX_DQ = 1.2
TAU_LIM = 900.0
DTAU_MAX = 140.0
FN = 0.6
ZETA = 1.0

RESID_REL_WARN = 0.05
SAT_WARN_RATIO = 0.10

THRUST_BOUND_SCALE = 100.0

# 조인트 궤적 생성 파라미터
VEL_MAX = 1.0          # rad/s (조인트 속도 상한)
ACC_MAX = 6.0          # rad/s^2 (조인트 가속도 상한)
VEL_NOISE_STD = 2.0    # 가속도 노이즈 표준편차 스케일(작을수록 더 부드러움)
VEL_DECAY = 0.9        # 속도 저역통과(1.0이면 유지, 0.0이면 즉시 감쇠)

TRAJ_MIN_POS_DELTA = 0.0   # (미사용) SE3 직접 샘플링 제거
TRAJ_MIN_ROT_DELTA = 0.0   # (미사용)


def _quiet_call(func, *args, **kwargs):
    with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return func(*args, **kwargs)


def _format_hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _progress_bar(done, total, prefix="Generating", start_time=None):
    bar_len = 40
    frac = done / total if total else 1.0
    bar = "#" * int(bar_len * frac) + "-" * (bar_len - int(bar_len * frac))
    msg = f"\r{prefix} [{bar}] {frac*100:5.1f}% ({done}/{total})"
    if start_time:
        elapsed = time.time() - start_time
        remain = (total - done) * elapsed / max(done, 1e-9)
        msg += f" | elapsed {_format_hms(elapsed)} | ETA {_format_hms(remain)}"
    print(msg, end="", flush=True)
    if done >= total:
        print()


def _is_finite(x):
    return np.all(np.isfinite(x))


def clip_step(vec, max_abs=None):
    if max_abs is None:
        return vec
    s = np.linalg.norm(vec, ord=np.inf)
    if s > max_abs:
        vec = vec * (max_abs / (s + 1e-12))
    return vec


def _rot_angle(Ra: R, Rb: R) -> float:
    dR = Ra.inv() * Rb
    return np.linalg.norm(dR.as_rotvec())


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


# ─────────────────────────────────────────────────────────
#   조인트 한계 읽기 (+ 폴백)
# ─────────────────────────────────────────────────────────
def get_joint_limits(model_param, dof):
    qmin = np.full((dof, 1), -np.pi)
    qmax = np.full((dof, 1),  np.pi)
    try:
        j = 0
        for odar in model_param["ODAR"]:
            jl = getattr(odar, "joint_limit", None)
            if jl is None:
                # body_joint_screw_axes 개수만큼 기본 한계 사용
                nj = len(odar.body_joint_screw_axes)
                j += nj
                continue
            # jl 형식: [(min, max), (min, max), ...]
            for (mn, mx) in jl:
                if j < dof:
                    qmin[j, 0] = float(mn)
                    qmax[j, 0] = float(mx)
                j += 1
    except Exception:
        pass
    return qmin, qmax


# ─────────────────────────────────────────────────────────
#   가능한(Feasible) 조인트 궤적 생성 (속도/가속/한계 보장)
# ─────────────────────────────────────────────────────────
def generate_feasible_joint_trajectory(dof, T, dt, qmin, qmax):
    rng = np.random.default_rng()
    # 초기값: 한계 안쪽 20% 여유
    mid = (qmin + qmax) / 2.0
    span = (qmax - qmin)
    q = np.zeros((T, dof, 1))
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)

    q0 = mid + (rng.random((dof, 1)) - 0.5) * 0.6 * span
    q0 = np.clip(q0, qmin + 0.2 * span, qmax - 0.2 * span)
    q[0] = q0
    dq[0] = np.zeros((dof, 1))
    ddq[0] = np.zeros((dof, 1))

    for t in range(1, T):
        # 랜덤 가속도 (노이즈 + 감쇠)
        noise = rng.normal(0.0, VEL_NOISE_STD, size=(dof, 1))
        # 가속도 제한
        acc = np.clip(noise, -ACC_MAX, ACC_MAX)

        # 속도 갱신 (저역통과)
        dq_t = VEL_DECAY * dq[t-1] + dt * acc
        dq_t = np.clip(dq_t, -VEL_MAX, VEL_MAX)

        # 위치 갱신
        q_t = q[t-1] + dt * dq_t

        # 한계 반사 처리
        for j in range(dof):
            if q_t[j, 0] < qmin[j, 0]:
                overflow = qmin[j, 0] - q_t[j, 0]
                q_t[j, 0] = qmin[j, 0] + overflow
                dq_t[j, 0] *= -0.6  # 경계에서 반발
            elif q_t[j, 0] > qmax[j, 0]:
                overflow = q_t[j, 0] - qmax[j, 0]
                q_t[j, 0] = qmax[j, 0] - overflow
                dq_t[j, 0] *= -0.6

        q[t] = q_t
        dq[t] = dq_t
        ddq[t] = (dq[t] - dq[t-1]) / dt

    return q, dq, ddq


# =========================================================
#               One sample generator (main loop)
# =========================================================
def generate_one_sample(link_count, T=1000, epsilon_scale=0.0, dt=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)

    base_param = get_parameters(link_count)
    base_param["ODAR"] = base_param["ODAR"][:link_count]

    screw_axes, inertias = [], []
    for odar in base_param["ODAR"]:
        screw_axes.extend(odar.body_joint_screw_axes)
        inertias.extend(odar.joint_inertia_tensor)

    base_param["LASDRA"].update(
        body_joint_screw_axes=screw_axes,
        inertia_matrix=inertias,
        dof=len(screw_axes),
    )
    model_param = _quiet_call(parameters_model, mode=0, params_prev=base_param)

    robot = LASDRA(model_param)
    fk_solver = ForwardKinematics(model_param)
    dof = model_param["LASDRA"]["dof"]

    # 조인트 한계 & 조인트 궤적 생성 (IK 대신)
    qmin, qmax = get_joint_limits(model_param, dof)
    q_des, dq_des, ddq_des = generate_feasible_joint_trajectory(dof, T, dt, qmin, qmax)

    # 컨트롤 & 외력 분배
    impedance_controller = ImpedanceControllerMaximal(model_param["ODAR"][-1])
    external_controller = ExternalActuation(model_param, robot)
    external_controller.apply_selective_mapping = True

    if hasattr(external_controller, "lp") and hasattr(external_controller, "qp"):
        if "ub" in external_controller.qp and "lb" in external_controller.qp:
            external_controller.qp["ub"] *= THRUST_BOUND_SCALE
            external_controller.qp["lb"] *= THRUST_BOUND_SCALE
        if "ub" in external_controller.lp and "lb" in external_controller.lp:
            external_controller.lp["ub"][:-1] *= THRUST_BOUND_SCALE
            external_controller.lp["lb"][:-1] *= THRUST_BOUND_SCALE

    for attr, val in [
        ("slack_weight", 1e4),
        ("slack_weight_lp", 1e4),
        ("ls_reg", 1e-3),
        ("rho", 1e-3),
        ("nullspace_damping", 1e-2),
    ]:
        if hasattr(external_controller, attr):
            try:
                setattr(external_controller, attr, val)
            except Exception:
                pass

    lam_dummy = np.zeros((T, 8 * link_count))
    _, type_matrix, label_matrix, t0, idx, which_mask, onset_idx = inject_faults(
        lam_dummy, epsilon_scale=epsilon_scale, return_labels=True
    )

    actual_q = np.zeros((T, dof, 1))
    actual_dq = np.zeros_like(actual_q)
    actual_q[0] = q_des[0]
    robot.set_joint_states(actual_q[0], np.zeros_like(actual_q[0]))

    desired_ee = np.zeros((T, 4, 4))
    actual_ee = np.zeros((T, 4, 4))

    desired_ee[0] = fk_solver.compute_end_effector_frame(q_des[0][:, 0])
    actual_ee[0] = fk_solver.compute_end_effector_frame(actual_q[0][:, 0])

    desired_link_rel = np.zeros((T, link_count, 4, 4))
    actual_link_rel = np.zeros((T, link_count, 4, 4))
    desired_link_cum = np.zeros((T, link_count, 4, 4))
    actual_link_cum = np.zeros((T, link_count, 4, 4))

    link_rel_des0, joint_counts = compute_link_relatives_pure(fk_solver, q_des[0][:, 0])
    link_rel_act0, _ = compute_link_relatives_pure(fk_solver, actual_q[0][:, 0])
    link_cum_des0 = compute_link_cumulative_from_rel(link_rel_des0)
    link_cum_act0 = compute_link_cumulative_from_rel(link_rel_act0)
    for k in range(link_count):
        desired_link_rel[0, k] = link_rel_des0[k]
        actual_link_rel[0, k] = link_rel_act0[k]
        desired_link_cum[0, k] = link_cum_des0[k]
        actual_link_cum[0, k] = link_cum_act0[k]

    dt = float(dt)
    tau_prev = np.zeros((dof,))  # rate-limit용

    for t in range(1, T):
        q = actual_q[t - 1]
        dq = actual_dq[t - 1]
        robot.set_joint_states(q, dq)

        Mjj = np.diag(robot.Mass)
        wn = 2 * np.pi * FN
        Kp_vec = (wn ** 2) * Mjj
        Kd_vec = (2 * ZETA * wn) * Mjj

        e = (q_des[t] - q).reshape(-1)
        de = (dq_des[t] - dq).reshape(-1)
        tau_raw = (Kp_vec * e + Kd_vec * de) + robot.Grav.reshape(-1)

        tau_raw = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
        dtau = np.clip(tau_raw - tau_prev, -DTAU_MAX, DTAU_MAX)
        tau = tau_prev + dtau
        tau_prev = tau.copy()

        if DEBUG_TAU_PRINT:
            print(t, float(np.linalg.norm(tau)), float(np.max(tau)), float(np.min(tau)))

        lam_cmd = external_controller.distribute_torque_lp(tau)

        lam_apply = lam_cmd.copy()
        faulty = (label_matrix[t] == 0)  # 0=고장, 1=정상
        if np.any(faulty):
            eps = float(epsilon_scale)
            ref = np.abs(lam_cmd) + 1e-9
            noise_fault = np.random.normal(0.0, eps, size=lam_cmd.shape) * ref
            scale_fault = 0.0  # stuck-off
            lam_apply[faulty] = scale_fault * lam_cmd[faulty] + noise_fault[faulty]

        for iL in range(link_count):
            thrust_i = lam_apply[8 * iL : 8 * (iL + 1)].reshape(-1, 1)
            robot.set_odar_body_wrench_from_thrust(thrust_i, iL)

        tau_odar = robot.get_joint_torque_from_odars()

        if CHECK_DISTRIBUTION_RESID:
            denom = np.linalg.norm(tau) + 1e-9
            resid_rel = float(np.linalg.norm(tau_odar - tau) / denom)
            sat_ratio = 0.0
            if hasattr(external_controller, "lp") and "ub" in external_controller.lp and "lb" in external_controller.lp:
                ub = np.asarray(external_controller.lp["ub"]).reshape(-1)
                lb = np.asarray(external_controller.lp["lb"]).reshape(-1)
                if ub.size == lam_cmd.size + 1 and lb.size == lam_cmd.size + 1:
                    ub = ub[:-1]
                    lb = lb[:-1]
                sat_ratio = float(np.mean((lam_cmd >= ub - 1e-9) | (lam_cmd <= lb + 1e-9)))
            if resid_rel > RESID_REL_WARN and sat_ratio > SAT_WARN_RATIO:
                print(f"[WARN] t={t} mapping residual={resid_rel:.2f} (consider tuning mapping/gains)")

        nxt = robot.get_next_joint_states(dt, tau_odar)
        qn, dqn = nxt["q"], np.clip(nxt["dq"], -10.0, 10.0)
        if not (_is_finite(qn) and _is_finite(dqn)):
            qn, dqn = q, dq

        actual_q[t], actual_dq[t] = qn, dqn
        robot.set_joint_states(qn, dqn)

        # FK로 desired/actual EE 및 링크 변환 기록
        desired_ee[t] = fk_solver.compute_end_effector_frame(q_des[t, :, 0])
        actual_ee[t] = fk_solver.compute_end_effector_frame(qn[:, 0])

        link_rel_des, _ = compute_link_relatives_pure(fk_solver, q_des[t, :, 0])
        link_rel_act, _ = compute_link_relatives_pure(fk_solver, qn[:, 0])
        link_cum_des = compute_link_cumulative_from_rel(link_rel_des)
        link_cum_act = compute_link_cumulative_from_rel(link_rel_act)

        for k in range(link_count):
            desired_link_rel[t, k] = link_rel_des[k]
            actual_link_rel[t, k] = link_rel_act[k]
            desired_link_cum[t, k] = link_cum_des[k]
            actual_link_cum[t, k] = link_cum_act[k]

    return (
        desired_ee,
        actual_ee,
        label_matrix,
        which_mask,
        onset_idx,
        int(t0),
        desired_link_rel,
        actual_link_rel,
        desired_link_cum,
        actual_link_cum,
        dof,
        np.array(joint_counts, dtype=np.int32),
    )


def _worker(args):
    link_count, T, epsilon_scale, dt, seed = args
    return generate_one_sample(
        link_count=link_count, T=T, epsilon_scale=epsilon_scale, dt=dt, seed=seed
    )


def generate_dataset_parallel(link_count, T, NUM_SAMPLES, dt, epsilon_scale, workers=None):
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1

    args_list = [(link_count, T, epsilon_scale, dt, 1000 + i) for i in range(NUM_SAMPLES)]

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
                (
                    d_ee,
                    a_ee,
                    l,
                    which_mask,
                    onset_idx,
                    t0,
                    d_lr,
                    a_lr,
                    d_lc,
                    a_lc,
                    dof,
                    joint_counts,
                ) = fut.result()

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
                    np.empty((0,)),
                    np.empty((0,)),
                    np.empty((0,)),
                    np.empty((0,)),
                    np.empty((0,)),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,)),
                    np.empty((0,)),
                    np.empty((0,)),
                    np.empty((0,)),
                    0,
                    np.empty((0,), dtype=np.int32),
                    True,
                )

            desired_ee = np.asarray(desired_ee_list)
            actual_ee = np.asarray(actual_ee_list)
            label = np.asarray(label_list)
            which_fault_mask = np.asarray(which_mask_list)
            onset_idx = np.asarray(onset_idx_list)
            t0_arr = np.asarray(t0_list, dtype=np.int32)
            d_link_rel = np.asarray(d_link_rel_list)
            a_link_rel = np.asarray(a_link_rel_list)
            d_link_cum = np.asarray(d_link_cum_list)
            a_link_cum = np.asarray(a_link_cum_list)
            dof_out = int(dof_list[0]) if dof_list else 0
            joint_counts_arr = np.asarray(joint_counts_list)

            return (
                desired_ee,
                actual_ee,
                label,
                which_fault_mask,
                onset_idx,
                t0_arr,
                d_link_rel,
                a_link_rel,
                d_link_cum,
                a_link_cum,
                dof_out,
                joint_counts_arr,
                True,
            )

    desired_ee = np.asarray(desired_ee_list)
    actual_ee = np.asarray(actual_ee_list)
    label = np.asarray(label_list)
    which_fault_mask = np.asarray(which_mask_list)
    onset_idx = np.asarray(onset_idx_list)
    t0_arr = np.asarray(t0_list, dtype=np.int32)
    d_link_rel = np.asarray(d_link_rel_list)
    a_link_rel = np.asarray(a_link_rel_list)
    d_link_cum = np.asarray(d_link_cum_list)
    a_link_cum = np.asarray(a_link_cum_list)
    dof_out = int(dof_list[0]) if dof_list else 0
    joint_counts_arr = np.asarray(joint_counts_list)

    return (
        desired_ee,
        actual_ee,
        label,
        which_fault_mask,
        onset_idx,
        t0_arr,
        d_link_rel,
        a_link_rel,
        d_link_cum,
        a_link_cum,
        dof_out,
        joint_counts_arr,
        False,
    )


if __name__ == "__main__":
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

    (
        desired_ee,
        actual_ee,
        label,
        which_mask,
        onset_idx,
        t0_arr,
        d_link_rel,
        a_link_rel,
        d_link_cum,
        a_link_cum,
        dof,
        joint_counts_arr,
        is_partial,
    ) = generate_dataset_parallel(link_count, T, NUM_SAMPLES, dt, epsilon_scale, workers)

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
            joint_counts=joint_counts_arr,
            desired_ee=desired_ee,
            actual_ee=actual_ee,
        )
        print(f"\nDataset saved successfully to {save_path}")

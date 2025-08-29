# -*- coding: utf-8 -*-
"""
data_generate_circle_faulty_dyn.py
 - 2-링크 원형 궤적 기반 단일 모터 고장 데이터 생성 (병렬 다중 샘플)
 - 저장 포맷: 두 번째 생성기(fault_dataset.npz)와 동일한 스키마
   keys:
     desired_link_rel, actual_link_rel, desired_link_cum, actual_link_cum,
     label (1=Fault, 0=Normal), which_fault_mask, onset_idx, t0, timestamps,
     dt, link_count, dof, joint_counts, desired_ee, actual_ee
"""

import os, sys, argparse, time, io, contextlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# threads 제한(선택)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynamics.forward_kinematics_class import ForwardKinematics
from dynamics.lasdra_class import LASDRA
from parameters import get_parameters
from parameters_model import parameters_model
from control.external_actuation import ExternalActuation

# ─────────────────────────────────────────────────────────
# 제어/분배 상수
# ─────────────────────────────────────────────────────────
FN = 0.6            # 임피던스 고유주파수(Hz)
ZETA = 1.0          # 감쇠비
TAU_LIM = 900.0     # 토크 절대 제한
DTAU_MAX = 140.0    # 토크 slew 제한
THRUST_BOUND_SCALE = 100.0  # 외력분배 bound 확장

# ── Traj utils (2-링크 평면) ──────────────────────────────────────────────────
def ik_2link_planar(l1, l2, x, y, elbow_up=True):
    r2 = x*x + y*y
    c2 = (r2 - l1*l1 - l2*l2) / (2.0*l1*l2)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(max(0.0, 1.0 - c2*c2))
    if not elbow_up: s2 = -s2
    q2 = np.arctan2(s2, c2)
    k1 = l1 + l2*c2; k2 = l2*s2
    q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return q1, q2

def build_circle_xy(radius=0.3, center=(0.6, 0.0), turns=1.0, T=1000):
    th = np.linspace(0, 2*np.pi*turns, T)
    x = center[0] + radius*np.cos(th)
    y = center[1] + radius*np.sin(th)
    return x, y

def compute_link_relatives_pure(fk_solver, q_vec):
    q_flat = np.asarray(q_vec).reshape(-1)
    link_T = []
    joint_counts = [len(odar.body_joint_screw_axes) for odar in fk_solver.ODAR]
    j0 = 0
    for k, nj in enumerate(joint_counts):
        T_link = np.eye(4)
        for j in range(nj):
            xi = fk_solver.Aset[:, j0 + j]
            T_link = T_link @ ForwardKinematics._exp_se3(xi, float(q_flat[j0 + j]))
        T_link = T_link @ fk_solver.T_joint_to_joint_set[k]
        link_T.append(T_link)
        j0 += nj
    return link_T, joint_counts

def compute_link_cumulative_from_rel(link_rel_list):
    out = []; T = np.eye(4)
    for Trel in link_rel_list:
        T = T @ Trel
        out.append(T.copy())
    return out

# ─────────────────────────────────────────────────────────
# 한 샘플 생성
# ─────────────────────────────────────────────────────────
def generate_one_circle_sample(T=1000, dt=0.01, radius=0.3, center=(0.6,0.0),
                               turns=1.0, elbow_up=True, motors_per_link=8, seed=None):
    if seed is not None:
        np.random.seed(seed)

    link_count = 2

    # ── 모델 준비
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
    model_param = parameters_model(mode=0, params_prev=base_param)
    robot = LASDRA(model_param)
    fk_solver = ForwardKinematics(model_param)
    dof = int(model_param["LASDRA"]["dof"])

    # 외력 분배기
    ext = ExternalActuation(model_param, robot)
    ext.apply_selective_mapping = True
    if hasattr(ext, "qp") and "ub" in ext.qp and "lb" in ext.qp:
        ext.qp["ub"] *= THRUST_BOUND_SCALE
        ext.qp["lb"] *= THRUST_BOUND_SCALE
    if hasattr(ext, "lp") and "ub" in ext.lp and "lb" in ext.lp:
        ext.lp["ub"][:-1] *= THRUST_BOUND_SCALE
        ext.lp["lb"][:-1] *= THRUST_BOUND_SCALE

    # ── 링크 길이
    q0 = np.zeros((dof,), dtype=float)
    link_rel0, joint_counts = compute_link_relatives_pure(fk_solver, q0)
    l1 = float(np.linalg.norm(link_rel0[0][:3, 3])); l2 = float(np.linalg.norm(link_rel0[1][:3, 3]))

    # ── 원하는 원 궤적 → q_des
    x, y = build_circle_xy(radius, center, turns, T)
    q_des = np.zeros((T, dof, 1), dtype=float)
    dq_des = np.zeros_like(q_des)

    for t in range(T):
        q1, q2 = ik_2link_planar(l1, l2, float(x[t]), float(y[t]), elbow_up=elbow_up)
        q_full = np.zeros((dof,), dtype=float)
        # 링크1, 링크2의 첫 조인트만 사용 (나머지 0)
        q_full[0] = q1
        if dof > 1:
            q_full[1] = q2
        q_des[t, :, 0] = q_full
        if t > 0:
            dq_des[t] = (q_des[t] - q_des[t-1]) / dt

    # ── 초기 상태
    actual_q  = np.zeros((T, dof, 1), dtype=float)
    actual_dq = np.zeros_like(actual_q)
    actual_q[0] = q_des[0]
    robot.set_joint_states(actual_q[0], np.zeros_like(actual_q[0]))

    # 포즈 버퍼 (두 번째 생성기 스키마와 동일)
    desired_link_rel = np.zeros((T, link_count, 4, 4))
    actual_link_rel  = np.zeros((T, link_count, 4, 4))
    desired_link_cum = np.zeros((T, link_count, 4, 4))
    actual_link_cum  = np.zeros((T, link_count, 4, 4))
    desired_ee = np.zeros((T, 4, 4))
    actual_ee  = np.zeros((T, 4, 4))

    # t=0 포즈 기록
    rel_des0, _ = compute_link_relatives_pure(fk_solver, q_des[0, :, 0])
    rel_act0, _ = compute_link_relatives_pure(fk_solver, actual_q[0, :, 0])
    cum_des0 = compute_link_cumulative_from_rel(rel_des0)
    cum_act0 = compute_link_cumulative_from_rel(rel_act0)
    for k in range(link_count):
        desired_link_rel[0, k] = rel_des0[k]; actual_link_rel[0, k] = rel_act0[k]
        desired_link_cum[0, k] = cum_des0[k]; actual_link_cum[0, k] = cum_act0[k]
    desired_ee[0] = cum_des0[-1]; actual_ee[0] = cum_act0[-1]

    # 라벨: 1=Fault, 0=Normal  (👉 시각화 스크립트와 일치)
    M = link_count * int(motors_per_link)
    label = np.zeros((T, M), dtype=np.float32)              # 기본 0(정상)
    # 고장 모터 선택 (단일 모터)
    fault_link = np.random.randint(0, link_count)
    fault_motor_in_link = np.random.randint(0, motors_per_link)
    gmi = fault_link * motors_per_link + fault_motor_in_link
    which_fault_mask = np.zeros((M,), dtype=np.uint8); which_fault_mask[gmi] = 1
    onset_idx = int(0.5 * T)  # 중간 이후 고장

    # 제어
    wn = 2.0 * np.pi * FN
    tau_prev = np.zeros((dof,), dtype=float)

    for t in range(1, T):
        q  = actual_q[t-1]
        dq = actual_dq[t-1]
        robot.set_joint_states(q, dq)

        # 임피던스 제어 토크
        Mjj = np.diag(robot.Mass)
        Kp_vec = (wn ** 2) * Mjj
        Kd_vec = (2.0 * ZETA * wn) * Mjj
        e  = (q_des[t]  - q ).reshape(-1)
        de = (dq_des[t] - dq).reshape(-1)
        tau_raw = (Kp_vec * e + Kd_vec * de) + robot.Grav.reshape(-1)

        # 토크 제한/슬루 제한
        tau_raw = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
        dtau = np.clip(tau_raw - tau_prev, -DTAU_MAX, DTAU_MAX)
        tau  = tau_prev + dtau
        tau_prev = tau.copy()

        # 외력 분배 → 람다
        lam_cmd = ext.distribute_torque_lp(tau)
        lam_apply = lam_cmd.copy()
        if t >= onset_idx:
            lam_apply[gmi] = 0.0           # stuck-off
            label[t, gmi] = 1.0            # ✅ 고장=1

        # 링크별 스러스터 적용
        for iL in range(link_count):
            seg = lam_apply[motors_per_link*iL : motors_per_link*(iL+1)].reshape(-1, 1)
            robot.set_odar_body_wrench_from_thrust(seg, iL)

        # 동역학 적분
        tau_odar = robot.get_joint_torque_from_odars()
        nxt = robot.get_next_joint_states(dt, tau_odar)
        qn  = nxt["q"]
        dqn = np.clip(nxt["dq"], -10.0, 10.0)
        if not (np.all(np.isfinite(qn)) and np.all(np.isfinite(dqn))):
            qn, dqn = q, dq

        actual_q[t], actual_dq[t] = qn, dqn
        robot.set_joint_states(qn, dqn)

        # 포즈 기록
        rel_des, _ = compute_link_relatives_pure(fk_solver, q_des[t, :, 0])
        rel_act, _ = compute_link_relatives_pure(fk_solver, qn[:, 0])
        cum_des = compute_link_cumulative_from_rel(rel_des)
        cum_act = compute_link_cumulative_from_rel(rel_act)
        for k in range(link_count):
            desired_link_rel[t, k] = rel_des[k]; actual_link_rel[t, k] = rel_act[k]
            desired_link_cum[t, k] = cum_des[k]; actual_link_cum[t, k] = cum_act[k]
        desired_ee[t] = cum_des[-1]; actual_ee[t] = cum_act[-1]

    # t0 (두 번째 코드 호환용) : 여기선 0으로
    t0 = 0

    return (desired_ee, actual_ee, label, which_fault_mask, onset_idx, t0,
            desired_link_rel, actual_link_rel, desired_link_cum, actual_link_cum,
            dof, np.array(joint_counts, dtype=np.int32))

# ─────────────────────────────────────────────────────────
# 워커
# ─────────────────────────────────────────────────────────
def _worker(args):
    return generate_one_circle_sample(*args)

# ─────────────────────────────────────────────────────────
# 병렬 생성
# ─────────────────────────────────────────────────────────
def generate_dataset_parallel(num_samples=20, T=1000, dt=0.01, motors_per_link=8,
                              radius=0.3, center=(0.6,0.0), turns=1.0, elbow_up=True,
                              workers=None):
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1

    args_list = [
        (T, dt, radius, center, turns, elbow_up, motors_per_link, 1000 + i)
        for i in range(num_samples)
    ]

    dEE_list, aEE_list, label_list = [], [], []
    mask_list, onset_list, t0_list = [], [], []
    dLR_list, aLR_list, dLC_list, aLC_list = [], [], [], []
    dof_list, jc_list = [], []

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, a) for a in args_list]
        for fut in as_completed(futs):
            (dEE, aEE, lab, mask, onset, t0, dLR, aLR, dLC, aLC, dof, jc) = fut.result()
            dEE_list.append(dEE); aEE_list.append(aEE)
            label_list.append(lab); mask_list.append(mask)
            onset_list.append(onset); t0_list.append(t0)
            dLR_list.append(dLR); aLR_list.append(aLR)
            dLC_list.append(dLC); aLC_list.append(aLC)
            dof_list.append(dof);  jc_list.append(jc)

    # stack
    desired_ee = np.asarray(dEE_list)         # [S,T,4,4]
    actual_ee  = np.asarray(aEE_list)         # [S,T,4,4]
    label      = np.asarray(label_list)       # [S,T,M]
    which_mask = np.asarray(mask_list)        # [S,M]
    onset_idx  = np.asarray(onset_list)       # [S]
    t0_arr     = np.asarray(t0_list, dtype=np.int32)  # [S]
    d_link_rel = np.asarray(dLR_list)         # [S,T,L,4,4]
    a_link_rel = np.asarray(aLR_list)         # [S,T,L,4,4]
    d_link_cum = np.asarray(dLC_list)         # [S,T,L,4,4]
    a_link_cum = np.asarray(aLC_list)         # [S,T,L,4,4]
    dof_out    = int(dof_list[0]) if dof_list else 0
    joint_counts_arr = np.asarray(jc_list)    # [S, link_count]

    return (desired_ee, actual_ee, label, which_mask, onset_idx, t0_arr,
            d_link_rel, a_link_rel, d_link_cum, a_link_cum,
            dof_out, joint_counts_arr)

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--T",       type=int, default=1000)
    ap.add_argument("--dt",      type=float, default=0.01)
    ap.add_argument("--radius",  type=float, default=0.30)
    ap.add_argument("--center_x", type=float, default=0.60)
    ap.add_argument("--center_y", type=float, default=0.00)
    ap.add_argument("--turns",   type=float, default=1.0)
    ap.add_argument("--elbow_up", type=int, default=1)
    ap.add_argument("--motors_per_link", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default=os.path.join("data_storage","link_2"))
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    link_count = 2
    timestamps = np.arange(args.T) * float(args.dt)

    (desired_ee, actual_ee, label, which_mask, onset_idx, t0_arr,
     d_link_rel, a_link_rel, d_link_cum, a_link_cum,
     dof_out, joint_counts_arr) = generate_dataset_parallel(
        num_samples=args.samples,
        T=args.T,
        dt=args.dt,
        motors_per_link=args.motors_per_link,
        radius=args.radius,
        center=(args.center_x, args.center_y),
        turns=args.turns,
        elbow_up=bool(args.elbow_up),
        workers=args.workers
    )

    save_path = os.path.join(args.save_dir, "circle_fault_dataset.npz")
    np.savez(
        save_path,
        desired_link_rel=d_link_rel,
        actual_link_rel=a_link_rel,
        desired_link_cum=d_link_cum,
        actual_link_cum=a_link_cum,
        label=label,                               # 1=Fault, 0=Normal
        which_fault_mask=which_mask,               # [S,M]
        onset_idx=onset_idx,                       # [S]
        t0=t0_arr,                                 # [S]
        timestamps=timestamps,                     # [T]
        dt=float(args.dt),
        link_count=int(link_count),
        dof=int(dof_out),
        joint_counts=joint_counts_arr,             # [S, link_count]
        desired_ee=desired_ee,
        actual_ee=actual_ee,
        motors_per_link=int(args.motors_per_link),
    )
    print(f"[OK] saved -> {save_path}")

"""
python3 data_generate/data_generate_circle_faulty_dyn.py \
  --samples 20 \
  --T 1000 \
  --dt 0.01 \
  --radius 0.3 \
  --center_x 0.6 --center_y 0.0 \
  --turns 1.0 \
  --motors_per_link 8 \
  --workers 0 \
  --save_dir ./data_storage/link_2
"""
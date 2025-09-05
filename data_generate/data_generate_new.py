# data_generate/data_generate_new.py
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
from scipy.sparse import block_diag as sp_block_diag

# ──────────────────── Project imports ────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dynamics.forward_kinematics_class import ForwardKinematics
from dynamics.lasdra_class import LASDRA
try:
    from fault_injection_fast import inject_faults_fast as inject_faults
except Exception:
    from fault_injection import inject_faults
from parameters import get_parameters
from parameters_model import parameters_model
from control.external_actuation import ExternalActuation      # ★ 외부 추력 분배기
from control.centralized_controller import CentralizedController  # ★ 중앙집중 PID (q→τ)

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

def clip_inf_norm(vec, max_abs=None):
    if max_abs is None: return vec
    s = np.linalg.norm(vec, ord=np.inf)
    return vec * (max_abs / (s + 1e-12)) if s > max_abs else vec

# ---------------- Math helpers ----------------
def random_unit():
    v = np.random.randn(3); n = np.linalg.norm(v)
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
# Desired SE(3) 궤적 생성기
# ------------------------------------------------------------
def generate_random_se3_series(
    fk_solver, q0, link_count, link_length, T=200,
    max_pos_step=0.004, max_rot_step=0.01, max_try=30,
):
    ℓ = float(link_length); n = link_count
    T0 = fk_solver.compute_end_effector_frame(q0.reshape(-1))
    p_cur = T0[:3, 3].copy()
    R_cur = R.from_matrix(T0[:3, :3])
    T_series = [T0]; r_max = n * ℓ

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

# ---------------- helpers: per-link transforms ----------------
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

# ---------------- CLIK & EE-PD → τ (optional 시각화/검증용) ----------------
def vee(mat):
    return np.array([
        mat[2,1] - mat[1,2],
        mat[0,2] - mat[2,0],
        mat[1,0] - mat[0,1]
    ], dtype=float)

def orientation_error(R_now, R_des):
    return 0.5 * vee(R_des.T @ R_now - R_now.T @ R_des)

def clik_step(fk, q, T_des, dt, k_pos=2.0, k_rot=2.0, damp=1e-3):
    """
    Closed-loop IK: v = [ω; v] = [k_rot*e_R ; k_pos*e_x], dq = J^+ v
    """
    q_flat = np.asarray(q).reshape(-1)          # (d,)
    T_now = fk.compute_end_effector_frame(q_flat)
    Rn, pn = T_now[:3,:3], T_now[:3,3]
    Rd, pd = T_des[:3,:3], T_des[:3,3]

    e_r = orientation_error(Rn, Rd).reshape(3,1)
    e_x = (pd - pn).reshape(3,1)
    v6  = np.vstack([k_rot*e_r, k_pos*e_x])     # desired body twist (6x1)

    J = np.asarray(fk.compute_end_effector_analytic_jacobian(q_flat), dtype=float)  # 6xd
    JJt = J @ J.T
    pinv = J.T @ np.linalg.inv(JJt + damp*np.eye(6))  # d×6

    dq = pinv @ v6                                   # (d,1)
    dq = clip_inf_norm(dq, 1.5)
    q_next_col = q_flat.reshape(-1,1) + dt*dq        # (d,1)
    return q_next_col.reshape(-1), v6, e_r, e_x, J   # 반환은 1D (d,)

def ee_pd_wrench(e_r, e_x, de_r, de_x,
                 Kp_r=np.diag([40,40,40]), Kd_r=np.diag([6,6,6]),
                 Kp_x=np.diag([900,900,900]), Kd_x=np.diag([90,90,90])):
    # F_des = [τ; f] (주의: D와 정렬) = [Kp_r e_r + Kd_r de_r ; Kp_x e_x + Kd_x de_x]
    tau = Kp_r @ e_r + Kd_r @ de_r
    f   = Kp_x @ e_x + Kd_x @ de_x
    return np.vstack([tau, f])

# ---------------- B_blkdiag helper (D와 정렬)
def build_B_blkdiag(params_model):
    """D가 [τ; f] 순서를 쓰므로, B(기본 [f; τ])에 swap(permutation) 적용"""
    eye_perm = np.block([
        [np.zeros((3, 3)), np.eye(3)],
        [np.eye(3), np.zeros((3, 3))]
    ])
    B_blocks = [eye_perm @ np.asarray(odar.B, dtype=float) for odar in params_model['ODAR']]
    return sp_block_diag(B_blocks).toarray()

# ---------------- Fault helpers ----------------
def first_fault_time_from_labels(label_matrix: np.ndarray) -> int:
    T = label_matrix.shape[0]
    for t in range(T):
        if np.any(label_matrix[t] == 0):
            return t
    return 10**9  # effectively "no fault"

def onset_vector_from_labels(label_matrix: np.ndarray) -> np.ndarray:
    T, M = label_matrix.shape
    onset = -np.ones((M,), dtype=np.int32)
    for j in range(M):
        idx = np.where(label_matrix[:, j] == 0)[0]
        if idx.size > 0:
            onset[j] = int(idx[0])
    return onset

# ---------------- One sample generator ----------------
def generate_one_sample(link_count, T=1000, epsilon_scale=0.0, dt=0.01, seed=None, bypass_Blambda=True):
    """
    생성 파이프라인:
      1) CLIK로 q_des(t) 궤적 생성
      2) CentralizedController 로 joint-space τ_joint(t) 산출 (q_des 추종)
      3) 외부 추력 분배기(전역 LP): D^T B λ = τ_joint  → λ(t)
      4) 실제 적용 토크: τ_apply = D^T B λ  (joint 외력 잔차 = 0 수렴)
      5) 고장(mask) 적용 후 동역학 적분
    """
    if seed is not None:
        np.random.seed(seed)

    # ----- 모델 파라미터 구성 -----
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
    fk_solver = robot.fk
    ext_act   = ExternalActuation(model_param, robot)                # ★ 전역 LP (D 포함)
    cc        = CentralizedController(model_param, robot)            # ★ 중앙집중 PID

    dof       = model_param['LASDRA']['dof']
    link_len  = float(model_param['ODAR'][0].length)

    # ----- 원하는 SE3 궤적 -----
    q0 = (np.random.rand(dof, 1) * 2 - 1) * 0.2 * np.pi
    T_des_series = generate_random_se3_series(
        fk_solver, q0, link_count, link_len, T=T,
        max_pos_step=0.001, max_rot_step=0.001
    )

    # ----- B_blkdiag (D와 순서 일치) -----
    B_blk = build_B_blkdiag(model_param)                 # (6L)×(8L)

    # ----- 고장 시나리오 -----
    lam_dummy = np.zeros((T, B_blk.shape[1]))
    _, type_matrix, label_matrix, t0, idx, which_mask, onset_idx_raw = inject_faults(
        lam_dummy, epsilon_scale=epsilon_scale, return_labels=True
    )
    label_matrix = np.asarray(label_matrix, dtype=np.int32)  # (T, 8*link_count)
    first_fault_t = first_fault_time_from_labels(label_matrix)
    onset_vec = onset_vector_from_labels(label_matrix)       # (8L,)

    # ----- 버퍼 -----
    q_des  = np.zeros((T, dof, 1)); dq_des = np.zeros_like(q_des)
    q_act  = np.zeros_like(q_des);   dq_act = np.zeros_like(q_des)

    q_des[0] = q0; q_act[0] = q0
    robot.set_joint_states(q_act[0], np.zeros_like(q_act[0]))
    cc.set_initial_state({"q": q_act[0], "dq": np.zeros_like(q_act[0])})

    desired_ee = np.zeros((T, 4, 4))
    actual_ee  = np.zeros((T, 4, 4))
    desired_ee[0] = fk_solver.compute_end_effector_frame(q_des[0][:,0])
    actual_ee[0]  = fk_solver.compute_end_effector_frame(q_act[0][:,0])

    desired_link_rel = np.zeros((T, link_count, 4, 4))
    actual_link_rel  = np.zeros((T, link_count, 4, 4))
    desired_link_cum = np.zeros((T, link_count, 4, 4))
    actual_link_cum  = np.zeros((T, link_count, 4, 4))

    lr0_des, joint_counts = compute_link_relatives_pure(fk_solver, q_des[0][:,0])
    lr0_act, _            = compute_link_relatives_pure(fk_solver, q_act[0][:,0])
    lc0_des = compute_link_cumulative_from_rel(lr0_des)
    lc0_act = compute_link_cumulative_from_rel(lr0_act)
    for k in range(link_count):
        desired_link_rel[0, k] = lr0_des[k]; desired_link_cum[0, k] = lc0_des[k]
        actual_link_rel[0, k]  = lr0_act[k];  actual_link_cum[0, k]  = lc0_act[k]

    # PD 미분항을 위한 이전 에러(선택적 로깅용)
    prev_e_r = np.zeros((3,1)); prev_e_x = np.zeros((3,1))

    # ----- 메인 루프 -----
    for t in range(1, T):
        # (1) CLIK로 q_des 업데이트
        T_des = T_des_series[t]
        qd_next, v6, e_r, e_x, J_des = clik_step(
            fk_solver, q_des[t-1][:,0], T_des, dt, k_pos=2.0, k_rot=2.0, damp=1e-4
        )
        q_des[t]  = qd_next.reshape(dof,1)
        dq_des[t] = ((q_des[t] - q_des[t-1]) / dt)

        # (2) Desired FK & per-link 기록
        desired_ee[t] = fk_solver.compute_end_effector_frame(q_des[t][:,0])
        lr_des, _ = compute_link_relatives_pure(fk_solver, q_des[t][:,0])
        lc_des    = compute_link_cumulative_from_rel(lr_des)
        for k in range(link_count):
            desired_link_rel[t, k] = lr_des[k]; desired_link_cum[t, k] = lc_des[k]

        # (2-1) EE-PD 렌치(선택적): 추적 성능 모니터만, 실제 토크 산출은 PID로 통일
        T_now_des = desired_ee[t]
        Rd, pd = T_des[:3,:3], T_des[:3,3]
        Rn, pn = T_now_des[:3,:3], T_now_des[:3,3]
        e_r = orientation_error(Rn, Rd).reshape(3,1)
        e_x = (pd - pn).reshape(3,1)
        de_r = (e_r - prev_e_r) / dt
        de_x = (e_x - prev_e_x) / dt
        prev_e_r, prev_e_x = e_r.copy(), e_x.copy()
        # F_des = ee_pd_wrench(e_r, e_x, de_r, de_x)  # 로그용이면 활성화

        # ------------------------ 핵심 변경: q→τ (PID) ------------------------
        # 목표 상태 세팅
        cc.set_desired_state({"q": q_des[t], "dq": dq_des[t]})
        # 현재 실제 상태에서 joint torque 요구량 계산
        tau_joint_cmd = cc.compute_tau(dt, q_act[t-1], dq_act[t-1])   # (d,1)
        tau_joint_cmd = clip_inf_norm(tau_joint_cmd, 5e4).reshape(-1)

        # (3) 고장 전 bypass: Desired==Actual 보장(학습 안정화용)
        if bypass_Blambda and (t < first_fault_t):
            q_act[t]  = q_des[t]
            dq_act[t] = dq_des[t]
            robot.set_joint_states(q_act[t], dq_act[t])
            actual_ee[t] = desired_ee[t]
            actual_link_rel[t] = desired_link_rel[t]
            actual_link_cum[t] = desired_link_cum[t]
            continue

        # --------------------- 외부 추력 전역 LP (MATLAB 동일) ---------------------
        # 현재 '실제 상태'에서의 D 사용을 위해, robot 상태는 q_act[t-1], dq_act[t-1] 로 유지
        lam_cmd = ext_act.distribute_joint_linf(tau_joint_cmd)  # solves D^T B λ = τ

        # (5) 라벨(고장) 적용
        lam_applied = lam_cmd.copy()
        row = label_matrix[t].reshape(-1)
        if row.size == lam_applied.size:
            lam_applied[row == 0] = 0.0

        # (6) 실제 τ 재계산 (같은 실제 상태의 A = DᵀB)
        D_now = np.asarray(robot.D, dtype=float)     # (6L×d)
        D_now_s = ext_act._sanitize_mat(D_now, clip=ext_act.MAT_CLIP)
        B_blk_s = ext_act._sanitize_mat(B_blk,  clip=ext_act.MAT_CLIP)
        Aeq_now = D_now_s.T @ B_blk_s  
        tau_apply = (Aeq_now @ lam_applied.reshape(-1,1)).reshape(dof,1)

        # (7) 동역학 적분 (현재 실제 상태에서)
        nxt = robot.get_next_joint_states(dt, tau_apply)
        qn, dqn = nxt['q'], np.clip(nxt['dq'], -10.0, 10.0)
        if not (_is_finite(qn) and _is_finite(dqn)):
            qn, dqn = q_act[t-1], dq_act[t-1]

        q_act[t], dq_act[t] = qn, dqn
        robot.set_joint_states(qn, dqn)

        # (8) 실제값 로깅
        actual_ee[t]  = fk_solver.compute_end_effector_frame(q_act[t][:,0])
        lr_act, _ = compute_link_relatives_pure(fk_solver, q_act[t][:,0])
        lc_act    = compute_link_cumulative_from_rel(lr_act)
        for k in range(link_count):
            actual_link_rel[t, k]  = lr_act[k]
            actual_link_cum[t, k]  = lc_act[k]

        # (9) (옵션) joint 외력 잔차 모니터: should → 0
        # resid = tau_apply.reshape(-1) - tau_joint_cmd
        # if np.linalg.norm(resid, ord=np.inf) > 1e-3:
        #     pass  # 필요 시 디버그 출력

    return (
        desired_ee, actual_ee, label_matrix, onset_vec, first_fault_t, int(t0),
        desired_link_rel, actual_link_rel, desired_link_cum, actual_link_cum,
        dof, np.array(joint_counts, dtype=np.int32)
    )

def _worker(args):
    link_count, T, epsilon_scale, dt, seed, bypass_Blambda = args
    return generate_one_sample(
        link_count=link_count, T=T,
        epsilon_scale=epsilon_scale, dt=dt, seed=seed, bypass_Blambda=bypass_Blambda
    )

def generate_dataset_parallel(link_count, T, NUM_SAMPLES, dt, epsilon_scale, workers=None, bypass_Blambda=True):
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1

    args_list = [(link_count, T, epsilon_scale, dt, 1000+i, bypass_Blambda) for i in range(NUM_SAMPLES)]

    desired_ee_list, actual_ee_list, label_list = [], [], []
    onset_vec_list, first_fault_t_list, t0_list = [], [], []
    d_link_rel_list, a_link_rel_list = [], []
    d_link_cum_list, a_link_cum_list = [], []
    dof_list, joint_counts_list = [], []

    print(f"Spawning {workers} worker(s)...", flush=True)
    done_cnt = 0; start_time = time.time()
    _progress_bar(done_cnt, NUM_SAMPLES, prefix="Generating samples", start_time=start_time)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        try:
            futs = [ex.submit(_worker, a) for a in args_list]
            for fut in as_completed(futs):
                (d_ee, a_ee, label_mat, onset_vec, first_fault_t, t0,
                 d_lr, a_lr, d_lc, a_lc, dof, joint_counts) = fut.result()

                desired_ee_list.append(d_ee)
                actual_ee_list.append(a_ee)
                label_list.append(label_mat)
                onset_vec_list.append(onset_vec)
                first_fault_t_list.append(first_fault_t)
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
                    np.empty((0,)), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32),
                    np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,)),
                    0, np.empty((0,), dtype=np.int32),
                    True
                )

            desired_ee = np.asarray(desired_ee_list)
            actual_ee  = np.asarray(actual_ee_list)
            label      = np.asarray(label_list)
            onset_vec_arr = np.asarray(onset_vec_list)
            first_fault_t_arr = np.asarray(first_fault_t_list, dtype=np.int32)
            t0_arr     = np.asarray(t0_list, dtype=np.int32)
            d_link_rel = np.asarray(d_link_rel_list)
            a_link_rel = np.asarray(a_link_rel_list)
            d_link_cum = np.asarray(d_link_cum_list)
            a_link_cum = np.asarray(a_link_cum_list)
            dof_out    = int(dof_list[0]) if dof_list else 0
            joint_counts_arr = np.asarray(joint_counts_list)
            return (desired_ee, actual_ee, label, onset_vec_arr, first_fault_t_arr, t0_arr,
                    d_link_rel, a_link_rel, d_link_cum, a_link_cum, dof_out, joint_counts_arr,
                    True)

    desired_ee = np.asarray(desired_ee_list)
    actual_ee  = np.asarray(actual_ee_list)
    label      = np.asarray(label_list)
    onset_vec_arr = np.asarray(onset_vec_list)
    first_fault_t_arr = np.asarray(first_fault_t_list, dtype=np.int32)
    t0_arr     = np.asarray(t0_list, dtype=np.int32)
    d_link_rel = np.asarray(d_link_rel_list)
    a_link_rel = np.asarray(a_link_rel_list)
    d_link_cum = np.asarray(d_link_cum_list)
    a_link_cum = np.asarray(a_link_cum_list)
    dof_out    = int(dof_list[0]) if dof_list else 0
    joint_counts_arr = np.asarray(joint_counts_list)
    return (desired_ee, actual_ee, label, onset_vec_arr, first_fault_t_arr, t0_arr,
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
        NUM_SAMPLES = 10
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
    try:
        ans = input("Bypass B·λ before first fault? (y/n, default y): ").strip().lower()
        bypass_Blambda = (ans != 'n')
    except Exception:
        bypass_Blambda = True

    save_dir = os.path.join("data_storage", f"link_{link_count}")
    os.makedirs(save_dir, exist_ok=True)

    dt = 0.01
    timestamps = np.arange(T) * dt

    (desired_ee, actual_ee, label, onset_vec_arr, first_fault_t_arr, t0_arr,
     d_link_rel, a_link_rel, d_link_cum, a_link_cum, dof, joint_counts_arr,
     is_partial) = generate_dataset_parallel(
        link_count, T, NUM_SAMPLES, dt, epsilon_scale, workers, bypass_Blambda=bypass_Blambda
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
            onset_idx=onset_vec_arr,
            first_fault_t=first_fault_t_arr,
            t0=t0_arr,
            timestamps=timestamps,
            dt=dt,
            link_count=link_count,
            dof=dof,
            joint_counts=joint_counts_arr,
            desired_ee=desired_ee,
            actual_ee=actual_ee
        )
        print(f"\nDataset saved successfully to {save_path}")

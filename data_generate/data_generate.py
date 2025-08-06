# data_generate/data_generate.py
import sys
import os
import io
import contextlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dynamics.forward_kinematics_class import ForwardKinematics
from dynamics.lasdra_class import LASDRA
from planning.closed_loop_inverse_kinematics import ClosedLoopInverseKinematics
from fault_injection import inject_faults
from parameters import get_parameters
from parameters_model import parameters_model
from scipy.spatial.transform import Rotation as R


# ----------------------------
# 유틸: 출력 억제 / 진행률
# ----------------------------
def _quiet_call(func, *args, **kwargs):
    """stdout/stderr를 잠시 막고 함수 호출 (내부 print 억제)."""
    with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return func(*args, **kwargs)

def _progress_bar(idx, total, prefix="Generating"):
    """간단한 진행률 바 (한 줄 업데이트)."""
    bar_len = 30
    frac = (idx + 1) / total
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix} [{bar}] {frac*100:5.1f}% ({idx+1}/{total})", end="", flush=True)
    if idx + 1 == total:
        print()


# ----------------------------
# IK 래퍼 (레포 시그니처 차이를 흡수)
# ----------------------------
def solve_ik(ik_solver, T_target, q_init):
    """
    ClosedLoopInverseKinematics의 반환 형식/메서드명을 유연 처리.
    우선: solve(T_target, q_init) → ndarray or dict({'q': ...})
    """
    if hasattr(ik_solver, "solve"):
        sol = ik_solver.solve(T_target, q_init)
    else:
        # FALLBACK: 메서드명이 다르면 아래에 맞게 수정
        # sol = ik_solver.run(T_target, q_init)
        raise AttributeError("ClosedLoopInverseKinematics has no method 'solve'")

    # 반환이 dict인 경우
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
    return q


def generate_random_se3_series(fk_solver, q0, link_count, link_length, T=200,
                               max_pos_step=0.001, max_rot_step=0.001):
    """
    링크 수에 따라 제약을 둔 무작위 SE(3) 시계열 생성기.
    - n = 1: 구면상 위치 + 고정 orientation
    - n = 2: 구 안 위치 + 제한된 z축 회전
    - n >=3: 구 안 위치 + 전방위 소규모 회전
    """
    T0 = fk_solver.compute_end_effector_frame(q0.reshape(-1))
    R_cur = R.from_matrix(T0[:3, :3])
    p_cur = T0[:3, 3].copy()
    T_series = [T0]

    r_max = link_count * float(link_length)

    for _ in range(1, T):
        for _try in range(10):
            # 1) 위치 증분
            dp = np.random.uniform(-max_pos_step, max_pos_step, size=3)
            p_cand = p_cur + dp

            # 2) 위치 보정 (링크 수별 제약)
            if link_count == 1:
                # 반지름 고정: 구면 위 유지
                norm = np.linalg.norm(p_cand)
                if norm > 1e-6:
                    p_cand = p_cand / norm * link_length
                else:
                    p_cand = np.array([link_length, 0, 0])
                R_new = R.identity()  # orientation 고정
            elif link_count == 2:
                # 거리만 제약 (반지름 2 이내)
                if np.linalg.norm(p_cand) > r_max:
                    dp *= 0.3
                    p_cand = p_cur + dp

                # 제한된 z축 회전
                axis = np.array([0.0, 0.0, 1.0])
                angle = np.random.uniform(-max_rot_step * 0.5, max_rot_step * 0.5)
                R_step = R.from_rotvec(axis * angle)
                R_new = R_cur * R_step
            else:
                # 일반 회전 및 위치 제약
                if np.linalg.norm(p_cand) > r_max:
                    dp *= 0.3
                    p_cand = p_cur + dp

                axis = np.random.randn(3)
                norm = np.linalg.norm(axis)
                if norm < 1e-12:
                    axis = np.array([1.0, 0.0, 0.0])
                else:
                    axis = axis / norm
                angle = np.random.uniform(-max_rot_step, max_rot_step)
                R_step = R.from_rotvec(axis * angle)
                R_new = R_cur * R_step

            # 3) SE(3) 생성
            T_new = np.eye(4)
            T_new[:3, :3] = R_new.as_matrix()
            T_new[:3, 3] = p_cand

            # 4) 유효성 검사
            if not np.any(np.isnan(T_new)) and not np.any(np.isinf(T_new)):
                T_series.append(T_new)
                R_cur = R_new
                p_cur = p_cand
                break
        else:
            T_series.append(T_series[-1].copy())

    return T_series


# ----------------------------
# 한 샘플 생성
# ----------------------------
def generate_one_sample(link_count, T=200, fault_time=100, epsilon_scale=0.05, dt=0.01):
    # 1) 파라미터 로드 및 집계
    base_param = get_parameters()
    base_param['LASDRA']['total_link_number'] = link_count
    base_param['ODAR'] = base_param['ODAR'][:link_count]

    screw_axes_all, inertia_all = [], []
    for odar in base_param['ODAR']:
        screw_axes_all.extend(odar.body_joint_screw_axes)   # (6,) 리스트
        inertia_all.extend(odar.joint_inertia_tensor)       # 6x6 리스트
    base_param['LASDRA']['body_joint_screw_axes'] = screw_axes_all
    base_param['LASDRA']['inertia_matrix'] = inertia_all
    base_param['LASDRA']['dof'] = len(screw_axes_all)

    # 이상적 모델 파라미터 (mode=0)
    model_param = _quiet_call(parameters_model, mode=0, params_prev=base_param)

    # 로봇/기구학/IK 객체
    robot = LASDRA(model_param)
    fk_solver = ForwardKinematics(model_param)
    ik_solver = ClosedLoopInverseKinematics(model_param)

    dof = model_param['LASDRA']['dof']

    # 링크 길이(수평 제약에 사용)
    try:
        link_len = float(model_param['ODAR'][0].length)
    except Exception:
        link_len = float(model_param['ODAR'][0]['length'])

    # 2) 임의 SE(3) 경로 생성
    q0 = (2 * np.random.rand(dof, 1) - 1) * np.pi
    T_des_series = generate_random_se3_series(
        fk_solver, q0, link_count, link_len, T=T, max_pos_step=0.001, max_rot_step=0.001
    )

    # 3) IK로 q_des, dq_des, ddq_des 생성
    q_des  = np.zeros((T, dof, 1))
    dq_des = np.zeros((T, dof, 1))
    ddq_des = np.zeros((T, dof, 1))

    try:
        q_des[0] = solve_ik(ik_solver, T_des_series[0], q0)
    except Exception:
        q_des[0] = q0  # IK 실패 시 초기값 유지

    for t in range(1, T):
        q_init = q_des[t-1]
        try:
            q_sol = solve_ik(ik_solver, T_des_series[t], q_init)
        except Exception:
            q_sol = q_init  # 실패 시 이전 값 유지
        q_des[t] = q_sol
        dq_des[t] = (q_des[t] - q_des[t-1]) / dt
        ddq_des[t] = (dq_des[t] - dq_des[t-1]) / dt

    ddq_des = np.clip(ddq_des, -20.0, 20.0)

    # 4) λ_des 계산
    lambda_des = np.zeros((T, 8 * link_count))
    for t in range(T):
        robot.set_joint_states(q_des[t], dq_des[t])
        tau = robot.Mass @ ddq_des[t] + robot.Cori @ dq_des[t] + robot.Grav  # (dof,1)
        H = robot.D.T @ robot.B_blkdiag                                     # (dof, 8N)
        lambda_des[t, :] = (np.linalg.pinv(H, rcond=1e-4) @ tau).reshape(-1)

    # 5) fault 주입
    lambda_faulty, type_matrix = inject_faults(
        lambda_des, fault_time=fault_time, epsilon_scale=epsilon_scale
    )  # lambda_faulty: (T, 8N), type_matrix: (N, 8) (1:정상,0:고장)

    # 6) λ_fault로 실제 궤적 적분 → actual SE(3)
    actual_q = np.zeros_like(q_des)
    actual_dq = np.zeros_like(dq_des)
    actual_q[0] = q_des[0]
    actual_dq[0] = np.zeros_like(q_des[0])

    robot.set_joint_states(actual_q[0], actual_dq[0])
    T_actual_series = [fk_solver.compute_end_effector_frame(actual_q[0, :, 0])]

    for t in range(1, T):
        # 링크별 외력 합성
        F_total = np.zeros((6 * link_count, 1))
        for i in range(link_count):
            thrust_i = lambda_faulty[t, i*8:(i+1)*8]        # (8,)
            Fi = robot.B_cell[i] @ thrust_i.reshape(-1, 1)  # (6,1)
            F_total[6*i:6*(i+1)] = Fi

        tau_fault = robot.D.T @ F_total                     # (dof,1)
        next_state = robot.get_next_joint_states(dt, tau_fault)
        actual_q[t] = next_state['q']
        actual_dq[t] = next_state['dq']
        robot.set_joint_states(actual_q[t], actual_dq[t])
        T_actual_series.append(fk_solver.compute_end_effector_frame(actual_q[t, :, 0]))

    # 7) 라벨 (200, 8N): t<fault_time → 1, t>=fault_time → type_matrix(펼침)
    label = np.ones((T, 8 * link_count), dtype=int)
    label_fault = type_matrix.reshape(-1)  # (8N,), 1정상 0고장
    if fault_time < T:
        label[fault_time:, :] = np.tile(label_fault, (T - fault_time, 1))

    desired_np = np.stack(T_des_series, axis=0)   # (T, 4, 4)
    actual_np  = np.stack(T_actual_series, axis=0)

    return desired_np, actual_np, label


# ----------------------------
# 메인: 여러 샘플 생성 및 저장
# ----------------------------
if __name__ == '__main__':
    link_count = int(input("How many links?: ").strip())

    os.makedirs("data_storage", exist_ok=True)

    T = 200
    FAULT_TIME = 100          # 0-based index: t>=100 → (1-based 101~200)
    NUM_SAMPLES = 50
    dt = 0.01                 # ✅ 고정 시간 간격(초)
    timestamps = np.arange(T) * dt  # ✅ [0.00, 0.01, ..., 1.99] (shape: (200,))

    all_desired, all_actual, all_labels = [], [], []

    try:
        for i in range(NUM_SAMPLES):
            _progress_bar(i, NUM_SAMPLES, prefix="Generating samples")
            desired, actual, label = generate_one_sample(
                link_count=link_count, T=T, fault_time=FAULT_TIME, epsilon_scale=0.05, dt=dt
            )
            all_desired.append(desired)  # (T,4,4)
            all_actual.append(actual)    # (T,4,4)
            all_labels.append(label)     # (T,8N)

        all_desired = np.array(all_desired)
        all_actual  = np.array(all_actual)
        all_labels  = np.array(all_labels)

        print("Saving to data_storage/fault_dataset.npz ...")
        np.savez("data_storage/fault_dataset.npz",
                 desired=all_desired, actual=all_actual, label=all_labels, timestamps=timestamps)  # ✅ 추가 저장
        print("Dataset saved successfully.")

    except KeyboardInterrupt:
        # 중단 시 부분 저장
        print("\nInterrupted. Saving partial results...")
        if len(all_desired) > 0:
            all_desired = np.array(all_desired)
            all_actual  = np.array(all_actual)
            all_labels  = np.array(all_labels)
            np.savez("data_storage/fault_dataset_partial.npz",
                     desired=all_desired, actual=all_actual, label=all_labels, timestamps=timestamps)
            print("Partial dataset saved to data_storage/fault_dataset_partial.npz")
        else:
            print("No samples were generated. Nothing to save.")

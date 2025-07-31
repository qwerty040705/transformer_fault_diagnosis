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
        print()  # 마지막엔 줄바꿈


def generate_random_trajectory(link_count, T=200, fault_time=100, epsilon_scale=0.05):
    base_param = get_parameters()
    base_param['LASDRA']['total_link_number'] = link_count
    base_param['ODAR'] = base_param['ODAR'][:link_count]

    screw_axes_all = []
    inertia_all = []
    for odar in base_param['ODAR']:
        # ODAR는 객체 속성 접근
        screw_axes_all.extend(odar.body_joint_screw_axes)
        inertia_all.extend(odar.joint_inertia_tensor)
    base_param['LASDRA']['body_joint_screw_axes'] = screw_axes_all
    base_param['LASDRA']['inertia_matrix'] = inertia_all
    base_param['LASDRA']['dof'] = len(screw_axes_all)

    # 내부 print 억제하여 모델 파라미터 로드
    model_param = _quiet_call(parameters_model, mode=0, params_prev=base_param)

    robot = LASDRA(model_param)
    fk_solver = ForwardKinematics(model_param)
    ik_solver = ClosedLoopInverseKinematics(model_param)  # (현재 예제에서는 사용 X)

    dof = model_param['LASDRA']['dof']
    q_des = np.zeros((T, dof, 1))
    dq_des = np.zeros((T, dof, 1))
    ddq_des = np.zeros((T, dof, 1))

    # 초기 상태
    q_des[0, :, 0] = (2 * np.random.rand(dof) - 1) * np.pi
    dq_des[0, :, 0] = 0
    ddq_des[0, :, 0] = 0

    dt = 0.1
    # 간단한 랜덤 워크로 q_des 생성
    for t in range(1, T):
        delta_q = (2 * np.random.rand(dof) - 1) * 0.1
        q_new = q_des[t-1, :, 0] + delta_q
        q_new = np.mod(q_new + np.pi, 2*np.pi) - np.pi

        T_new = fk_solver.compute_end_effector_frame(q_new)
        x, y, z = T_new[0, 3], T_new[1, 3], T_new[2, 3]
        attempts = 0
        while (z < -1 or z > 3 or np.hypot(x, y) > link_count * 1.0) and attempts < 5:
            delta_q = -0.5 * delta_q
            q_new = q_des[t-1, :, 0] + delta_q
            q_new = np.mod(q_new + np.pi, 2*np.pi) - np.pi
            T_new = fk_solver.compute_end_effector_frame(q_new)
            x, y, z = T_new[0, 3], T_new[1, 3], T_new[2, 3]
            attempts += 1
        if z < -1 or z > 3 or np.hypot(x, y) > link_count * 1.0:
            q_new = q_des[t-1, :, 0]

        q_des[t, :, 0] = q_new
        dq_des[t, :, 0] = (q_des[t, :, 0] - q_des[t-1, :, 0]) / dt
        ddq_des[t, :, 0] = (dq_des[t, :, 0] - dq_des[t-1, :, 0]) / dt if t > 1 else 0

    # 목표 T 시퀀스
    T_des_series = [fk_solver.compute_end_effector_frame(q_des[t, :, 0]) for t in range(T)]

    # 추력 분배 해 (정상 상태) 계산
    lambda_des = np.zeros((T, 8 * link_count))
    for t in range(T):
        robot.set_joint_states(q_des[t], dq_des[t])
        tau = robot.Mass @ ddq_des[t] + robot.Cori @ dq_des[t] + robot.Grav  # (dof,1)
        tau = tau.flatten()
        H = robot.D.T @ robot.B_blkdiag  # (dof, sum_rotors)
        lambda_sol = np.linalg.pinv(H) @ tau
        lambda_des[t, :] = lambda_sol

    # 고장 주입 (2D 입력 지원)
    lambda_faulty, type_matrix = inject_faults(lambda_des, fault_time=fault_time, epsilon_scale=epsilon_scale)

    # 실제 궤적 적분
    actual_q = np.zeros_like(q_des)
    actual_dq = np.zeros_like(dq_des)
    actual_q[0] = q_des[0]
    actual_dq[0] = dq_des[0]
    robot.set_joint_states(actual_q[0], actual_dq[0])
    T_actual_series = [fk_solver.compute_end_effector_frame(actual_q[0, :, 0])]
    for t in range(1, T):
        F_total = np.zeros((6 * link_count, 1))
        for link_idx in range(link_count):
            thrust_i = lambda_faulty[t, link_idx*8:(link_idx+1)*8]
            Fi = robot.B_cell[link_idx] @ thrust_i.reshape(-1, 1)
            F_total[6*link_idx:6*(link_idx+1)] = Fi
        tau_fault = robot.D.T @ F_total  # (dof,1)
        next_state = robot.get_next_joint_states(dt, tau_fault)
        actual_q[t] = next_state['q']
        actual_dq[t] = next_state['dq']
        robot.set_joint_states(actual_q[t], actual_dq[t])
        T_actual_series.append(fk_solver.compute_end_effector_frame(actual_q[t, :, 0]))

    return T_des_series, T_actual_series, type_matrix


if __name__ == '__main__':
    link_count = int(input("How many links?: "))

    os.makedirs("data_storage", exist_ok=True)

    NUM_SAMPLES = 1000
    all_desired = []
    all_actual = []
    all_labels = []

    try:
        for i in range(NUM_SAMPLES):
            # 진행률 표시
            _progress_bar(i, NUM_SAMPLES, prefix="Generating samples")

            T_des, T_act, label = generate_random_trajectory(link_count=link_count)
            all_desired.append(np.stack(T_des))
            all_actual.append(np.stack(T_act))
            all_labels.append(label)

        # 배열로 변환
        all_desired = np.array(all_desired)
        all_actual = np.array(all_actual)
        all_labels = np.array(all_labels)

        print("Saving to fault_dataset.npz...")
        np.savez("data_storage/fault_dataset.npz", desired=all_desired, actual=all_actual, label=all_labels)
        print("Dataset saved successfully.")

    except KeyboardInterrupt:
        # CTRL+C 시, 생성한 만큼이라도 저장
        print("\nInterrupted. Saving partial results...")
        if len(all_desired) > 0:
            all_desired = np.array(all_desired)
            all_actual = np.array(all_actual)
            all_labels = np.array(all_labels)
            np.savez("data_storage/fault_dataset_partial.npz", desired=all_desired, actual=all_actual, label=all_labels)
            print("Partial dataset saved to data_storage/fault_dataset_partial.npz")
        else:
            print("No samples were generated. Nothing to save.")

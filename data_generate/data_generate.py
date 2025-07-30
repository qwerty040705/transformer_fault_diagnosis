'''
아직 수정중
'''

import os
import numpy as np

from planning.closed_loop_inverse_kinematics import generate_desired_trajectory
from dynamics.forward_kinematics_class import ForwardKinematics
from dynamics.lasdra_class import LASDRA
from fault_injection import inject_faults
from parameters import get_parameters
from parameters_model import parameters_model

# 저장 경로 설정
save_dir = "/home/cdb/transformer_fault_diagnosis/data_generate/data_storage"
os.makedirs(save_dir, exist_ok=True)

# 샘플 개수
NUM_SAMPLES = 10

# 시스템 초기화
param = get_parameters()
model_param = parameters_model()
robot = LASDRA(param, model_param)
fk_solver = ForwardKinematics(param)

for i in range(NUM_SAMPLES):
    # 1. Desired trajectory 생성
    T_d, q_d = generate_desired_trajectory(param)

    # 2. Inverse Dynamics로 lambda_d 계산
    lambda_d = robot.inverse_motor_dynamics(q_d)

    # 3. 고장 주입
    lambda_a_flat, type_matrix = inject_faults(lambda_d.flatten())
    lambda_a = lambda_a_flat.reshape(lambda_d.shape)

    # 4. 고장난 lambda로 실제 q_a 계산
    q_a = robot.forward_dynamics(lambda_a)

    # 5. 실제 q_a로 T_a 계산
    T_a = np.array([fk_solver.compute(q_a[:, t]) for t in range(q_a.shape[1])])

    # 6. 저장
    save_path = os.path.join(save_dir, f"sample_{i:03d}.npz")
    np.savez_compressed(
        save_path,
        T_d=T_d, q_d=q_d, lambda_d=lambda_d,
        T_a=T_a, q_a=q_a, lambda_a=lambda_a,
        type_matrix=type_matrix
    )

    print(f"[{i+1}/{NUM_SAMPLES}] Saved: {save_path}")

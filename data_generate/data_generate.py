'''
아직 수정중
'''

import os
import numpy as np

from planning.closed_loop_inverse_kinematics import ClosedLoopInverseKinematics
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

    # 1. Desired trajectory 생성

    # 2. Inverse Dynamics로 lambda_d 계산

    # 3. 고장 주입


    # 4. 고장난 lambda로 실제 q_a 계산

    # 5. 실제 q_a로 T_a 계산

    # 6. 저장

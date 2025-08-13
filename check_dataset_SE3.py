import os
import numpy as np

link_count = int(input("How many links do you want to check?: ").strip())
data_dir = os.path.join("data_storage", f"link_{link_count}")
data_path = os.path.join(data_dir, "fault_dataset.npz")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found.")

data = np.load(data_path)
desired = data['desired']  # (S, 200, 4, 4)
actual = data['actual']    # (S, 200, 4, 4)
label = data['label']      # (S, 200, 8*N)

S, T, _, _ = desired.shape
motor_count = label.shape[2]

print(f"desired.shape = {desired.shape}")
print(f"actual.shape  = {actual.shape}")
print(f"label.shape   = {label.shape} (motor_count={motor_count})")


sample_idx = 0
time_idx_list = [0, 1,10,50,100,150,200,250, 500, 750, 999] 

def decompose_se3(T):
    R = T[:3, :3] 
    p = T[:3, 3]   
    return R, p

for time_idx in time_idx_list:
    print("\n" + "="*50)
    print(f"[Sample {sample_idx}, Time {time_idx}]")

    # 라벨 출력
    print("Motor labels:", label[sample_idx, time_idx])

    # SE(3) 행렬 출력
    print("\nDesired SE(3):\n", desired[sample_idx, time_idx])
    print("\nActual SE(3):\n", actual[sample_idx, time_idx])

    # 위치 + 회전 분리 출력
    R_des, p_des = decompose_se3(desired[sample_idx, time_idx])
    R_act, p_act = decompose_se3(actual[sample_idx, time_idx])

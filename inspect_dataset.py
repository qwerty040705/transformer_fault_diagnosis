import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
data = np.load('data_storage/fault_dataset.npz')
desired = data['desired']     # shape: (S, 200, 4, 4)
actual = data['actual']       # shape: (S, 200, 4, 4)
label = data['label']         # shape: (S, 200, 8*N)

print("✅ Data loaded.")
print(" - desired.shape:", desired.shape)
print(" - actual.shape:", actual.shape)
print(" - label.shape:", label.shape)

# NaN 또는 inf 검사
for name, array in [('desired', desired), ('actual', actual), ('label', label)]:
    if np.isnan(array).any():
        print(f"❌ {name}에 NaN이 포함되어 있습니다.")
    elif np.isinf(array).any():
        print(f"❌ {name}에 inf가 포함되어 있습니다.")
    else:
        print(f"✅ {name}에는 NaN/inf가 없습니다.")

# 시각화: 첫 번째 샘플의 궤적 (End-effector position)
def extract_position(T_series):
    """(200, 4, 4) → (200, 3) position only"""
    return T_series[:, :3, 3]

sample_idx = 0
des_pos = extract_position(desired[sample_idx])
act_pos = extract_position(actual[sample_idx])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(des_pos, label=["x", "y", "z"])
plt.title("Desired Trajectory (Position)")
plt.xlabel("Time step"); plt.ylabel("Position [m]")
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(act_pos, label=["x", "y", "z"])
plt.title("Actual Trajectory (Position)")
plt.xlabel("Time step"); plt.ylabel("Position [m]")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

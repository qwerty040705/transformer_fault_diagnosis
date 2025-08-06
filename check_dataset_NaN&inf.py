import os
import numpy as np

link_count = int(input("How many links do you want to check?: ").strip())

data_dir = os.path.join("data_storage", f"link_{link_count}")
data_path = os.path.join(data_dir, "fault_dataset.npz")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ {data_path} 파일이 존재하지 않습니다.")

data = np.load(data_path)
desired = data['desired']     # shape: (S, 200, 4, 4)
actual = data['actual']       # shape: (S, 200, 4, 4)
label = data['label']         # shape: (S, 200, 8*N)

print("✅ Data loaded from:", data_path)
print(" - desired.shape:", desired.shape)
print(" - actual.shape:", actual.shape)
print(" - label.shape:", label.shape)


for name, array in [('desired', desired), ('actual', actual), ('label', label)]:
    if np.isnan(array).any():
        print(f"❌ {name}에 NaN이 포함되어 있습니다.")
    elif np.isinf(array).any():
        print(f"❌ {name}에 inf가 포함되어 있습니다.")
    else:
        print(f"✅ {name}에는 NaN/inf가 없습니다.")

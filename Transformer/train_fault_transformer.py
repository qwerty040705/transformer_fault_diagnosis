import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from fault_diagnosis_model import FaultDiagnosisTransformer

# -------- Config --------
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    link_count = int(input("How many links?: ").strip())
except Exception:
    link_count = 1
    print("[WARN] Invalid input. Fallback to link_count=1")

data_path = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")

batch_size = 16
epochs = 200
lr = 1e-3
weight_decay = 1e-4
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📥 Using device: {device}")

# Early Stopping
patience = 10
patience_counter = 0

torch.manual_seed(seed)
np.random.seed(seed)

# -------- Load dataset --------
data = np.load(data_path)
desired = data["desired"]    # (S, T, 4, 4)
actual = data["actual"]      # (S, T, 4, 4)
labels = data["label"]       # (S, T, M)  where M = 8 * link_count

S, T, _, _ = desired.shape
M = labels.shape[2]
print(f"📦 Loaded: S={S}, T={T}, M={M} (motors) from link_{link_count}")

# -------- Build 24-dim inputs: top 3x4 (12) for desired/actual --------
des_12 = desired[:, :, :3, :4].reshape(S, T, 12)  # (S, T, 12)
act_12 = actual[:, :, :3, :4].reshape(S, T, 12)   # (S, T, 12)
X = np.concatenate([des_12, act_12], axis=2).astype(np.float32)  # (S, T, 24)
y = labels[:, -1, :].astype(np.float32)  # (S, M)

# -------- Initial Dataset and Split --------
dataset_all = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
train_size = int(0.8 * S)
val_size = S - train_size
train_ds, val_ds = random_split(dataset_all, [train_size, val_size],
                                generator=torch.Generator().manual_seed(seed))

# -------- Standardization (fit on train only) --------
X_train = train_ds.dataset.tensors[0][train_ds.indices].numpy()  # (train_size, T, 24)
train_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
train_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-6

def apply_norm(tensor, mean, std):
    x = tensor.numpy()
    x = (x - mean) / std
    return torch.from_numpy(x.astype(np.float32))

# Normalize and move to device
X_norm = apply_norm(dataset_all.tensors[0], train_mean, train_std).to(device)
y_tensor = dataset_all.tensors[1].to(device)
dataset_all = TensorDataset(X_norm, y_tensor)
train_ds, val_ds = random_split(dataset_all, [train_size, val_size],
                                generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Rough stats (optional)
X_mean_full = X.reshape(-1, X.shape[-1]).mean(axis=0)
X_std_full = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
print("📊 Rough global mean/std (pre-split):", X_mean_full[:4], X_std_full[:4])

# -------- Model / Loss / Optimizer --------
model = FaultDiagnosisTransformer(input_dim=24, output_dim=M, max_seq_len=T).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# -------- Train loop with Early Stopping --------
best_val = float("inf")
ckpt_dir = os.path.join(repo_root, "Transformer")
os.makedirs(ckpt_dir, exist_ok=True)
save_path = os.path.join(ckpt_dir, f"Transformer_link_{link_count}.pth")

for ep in range(1, epochs + 1):
    # Train
    model.train()
    tr_loss_sum = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)  # xb, yb are already on device
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss_sum += loss.item() * xb.size(0)
    tr_loss = tr_loss_sum / max(1, len(train_ds))

    # Validation
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss_sum += loss.item() * xb.size(0)
    val_loss = val_loss_sum / max(1, len(val_ds))

    print(f"🏋️ [{ep:03d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

    # Early stopping check
    if val_loss < best_val - 1e-6:
        best_val = val_loss
        torch.save({
            "model_state": model.state_dict(),
            "train_mean": train_mean,
            "train_std": train_std,
            "input_dim": 24,
            "T": T,
            "M": M,
            "cfg": dict(d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1)
        }, save_path)
        print(f"💾 Saved best model to {save_path}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"⏳ No improvement. Patience counter: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("🛑 Early stopping triggered.")
            break

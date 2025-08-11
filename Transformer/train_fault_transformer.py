import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from fault_diagnosis_model import FaultDiagnosisTransformer 

# ─── Config ─────────────────────────────────────────────
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    link_count = int(input("How many links?: ").strip())
except Exception:
    link_count = 1
    print("[WARN] Invalid input. Fallback to link_count=1")

data_path   = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")
batch_size  = 16
lr, wd      = 1e-3, 1e-4
seed        = 42
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📥 device:", device)

torch.manual_seed(seed); np.random.seed(seed)

# ─── Load dataset ───────────────────────────────────────
data    = np.load(data_path)
desired = data["desired"]          # (S,T,4,4)
actual  = data["actual"]
labels  = data["label"]            # (S,T,M)

S, T, _, _ = desired.shape
M = labels.shape[2]
epochs = int(0.8 * S)
print(f"Loaded S={S}, T={T}, M={M} | epochs={epochs}")

# ─── Build inputs (S,T,24) ──────────────────────────────
des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)
X = np.concatenate([des_12, act_12], axis=2).astype(np.float32)   # (S,T,24)
y = labels.astype(np.float32)                                     # (S,T,M)

# ─── Dataset & split ────────────────────────────────────
full_ds   = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
train_sz  = int(0.8 * S)
val_sz    = S - train_sz
train_ds, val_ds = random_split(full_ds, [train_sz, val_sz],
                                generator=torch.Generator().manual_seed(seed))

# ─── Standardize (fit on train only) ────────────────────
X_train = train_ds.dataset.tensors[0][train_ds.indices].numpy()
μ = X_train.reshape(-1, 24).mean(0)
σ = X_train.reshape(-1, 24).std(0) + 1e-6
norm = lambda a: torch.from_numpy(((a.numpy() - μ) / σ).astype(np.float32))

X_norm = norm(full_ds.tensors[0]).to(device)
y_all  = full_ds.tensors[1].to(device)
dataset_all = TensorDataset(X_norm, y_all)
train_ds, val_ds = random_split(dataset_all, [train_sz, val_sz],
                                generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# ─── Model / loss / optim ───────────────────────────────
model = FaultDiagnosisTransformer(
    input_dim=24,
    output_dim=M,
    max_seq_len=T,
    d_model=64,
    nhead=8,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1
).to(device)

loss_fn = nn.BCEWithLogitsLoss()
opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

# ─── Train loop ─────────────────────────────────────────
ckpt_dir = os.path.join(repo_root, "Transformer"); os.makedirs(ckpt_dir, exist_ok=True)
save_path = os.path.join(ckpt_dir, f"Transformer_link_{link_count}.pth")

for ep in range(1, epochs + 1):
    # Train
    model.train()
    train_loss_sum = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)                  # (B,T,M)
        loss   = loss_fn(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_loss_sum += loss.item() * xb.size(0)
    tr_loss = train_loss_sum / len(train_ds)
    
    # Validation phase
    model.eval()
    val_loss_sum = 0.0
    tp = fp = fn = 0  
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # (B,T,M)
            val_loss_sum += loss_fn(logits, yb).item() * xb.size(0)
            pred = (torch.sigmoid(logits) >= 0.5).int()  # (B,T,M) 0 or 1
            yb_int = yb.int()
            for j in range(xb.size(0)):
                y_np = yb_int[j].cpu().numpy()
                p_np = pred[j].cpu().numpy()
                for m in range(M):
                    if 0 in y_np[:, m]:
                        if 0 in p_np[:, m]:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if 0 in p_np[:, m]:
                            fp += 1
    val_loss = val_loss_sum / len(val_ds)
    # Precision, Recall, F1 계산
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * tp / (2 * tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0

    print(f"[{ep:03d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} "
          f"| Det Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")

# ─── Save ───────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "train_mean" : μ, "train_std": σ,
    "input_dim": 24, "T": T, "M": M,
    "cfg": dict(d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1)
}, save_path)
print("✅ saved:", save_path)

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from fault_diagnosis_model import FaultDiagnosisTransformer

# ---------------- Config ----------------
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 64
seed       = 42

# ---------------- I/O Paths -------------
try:
    link_count = int(input("How many links?: ").strip())
except Exception:
    link_count = 1
    print("[WARN] Invalid input. Fallback to link_count=1")

data_path = os.path.join(
    repo_root, f"data_storage/link_{link_count}/fault_dataset.npz"
)
ckpt_path = os.path.join(
    repo_root, "Transformer", f"Transformer_link_{link_count}.pth"
)

# ---------------- Load Data -------------
data    = np.load(data_path)
desired = data["desired"]          # (S, T, 4, 4)
actual  = data["actual"]           # (S, T, 4, 4)
labels  = data["label"]            # (S, T, M)

S, T, _, _ = desired.shape
M = labels.shape[2]

des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)
X = np.concatenate([des_12, act_12], axis=2).astype(np.float32)   # (S,T,24)
y = labels[:, -1, :].astype(np.float32)                           # (S,M)

# ------------- Load Checkpoint ----------
ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
train_mean  = ckpt["train_mean"]
train_std   = ckpt["train_std"]
cfg         = ckpt["cfg"]

assert ckpt["input_dim"] == 24 and ckpt["T"] == T and ckpt["M"] == M, \
    "Checkpoint/Data shape mismatch."

X = (X - train_mean) / train_std
X_t = torch.from_numpy(X)          
y_t = torch.from_numpy(y)

# Same split recreation
dataset_all = TensorDataset(X_t, y_t)
train_size  = int(0.8 * S)
val_size    = S - train_size
_, val_ds = random_split(
    dataset_all, [train_size, val_size],
    generator=torch.Generator().manual_seed(seed)
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False  
)

# ------------- Rebuild Model ------------
model = FaultDiagnosisTransformer(
    input_dim=24,
    d_model=cfg["d_model"],
    nhead=cfg["nhead"],
    num_layers=cfg["num_layers"],
    dim_feedforward=cfg["dim_feedforward"],
    dropout=cfg["dropout"],
    output_dim=M,
    max_seq_len=T,
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ------------- Evaluation ---------------
all_probs, all_true = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        probs = torch.sigmoid(model(xb)).cpu().numpy()
        all_probs.append(probs)
        all_true.append(yb.cpu().numpy())

probs = np.concatenate(all_probs, axis=0)   # (N, M)
true  = np.concatenate(all_true,  axis=0)   # (N, M)

# AUROC / AUPRC
try:
    auroc_macro = roc_auc_score(true, probs, average="macro")
    auroc_micro = roc_auc_score(true, probs, average="micro")
except ValueError:
    auroc_macro = np.nan
    auroc_micro = np.nan

auprc_macro = average_precision_score(true, probs, average="macro")
auprc_micro = average_precision_score(true, probs, average="micro")

# F1@0.5
pred      = (probs >= 0.5).astype(np.float32)
f1_macro  = f1_score(true, pred, average="macro", zero_division=0)
f1_micro  = f1_score(true, pred, average="micro", zero_division=0)
per_motor_f1 = [f1_score(true[:, m], pred[:, m], zero_division=0) for m in range(M)]

# ------------- Print --------------------
print("\n==== Validation Metrics ====")
print(f"AUROC  macro: {auroc_macro:.4f} | micro: {auroc_micro:.4f}")
print(f"AUPRC  macro: {auprc_macro:.4f} | micro: {auprc_micro:.4f}")
print(f"F1@0.5 macro: {f1_macro:.4f} | micro: {f1_micro:.4f}")
print("Per-motor F1:", " ".join([f"{v:.3f}" for v in per_motor_f1]))

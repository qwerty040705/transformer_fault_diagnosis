import os, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from fault_diagnosis_model import FaultDiagnosisTransformer  

# ─── Config ────────────────────────────────────────────
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64; seed = 42
print("device:", device)

# ─── I/O paths ─────────────────────────────────────────
link_count = int(input("How many links?: ") or 1)
data_path  = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")
ckpt_path  = os.path.join(repo_root, "Transformer", f"Transformer_link_{link_count}.pth")

# ─── Load data ─────────────────────────────────────────
data    = np.load(data_path)
desired = data["desired"]          # (S,T,4,4)
actual  = data["actual"]
labels  = data["label"]            # (S,T,M)

S, T, _, _ = desired.shape; M = labels.shape[2]

des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)
X  = np.concatenate([des_12, act_12], axis=2).astype(np.float32)   # (S,T,24)
y  = labels.astype(np.float32)                                     # (S,T,M)

# ─── Load checkpoint ──────────────────────────────────
ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
mean, std  = ckpt["train_mean"], ckpt["train_std"]
cfg        = ckpt["cfg"]
assert (ckpt["input_dim"], ckpt["T"], ckpt["M"]) == (24, T, M), "shape mismatch"

# normalize
X = (X - mean) / std
ds_all = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

# split (0.8 / 0.2) 동일 seed
train_sz = int(0.8*S); val_sz = S - train_sz
_, val_ds = random_split(ds_all, [train_sz, val_sz],
                         generator=torch.Generator().manual_seed(seed))
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# ─── rebuild model ────────────────────────────────────
model = FaultDiagnosisTransformer(
    input_dim=24,
    d_model=cfg["d_model"], nhead=cfg["nhead"],
    num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
    dropout=cfg["dropout"], output_dim=M, max_seq_len=T
).to(device)
model.load_state_dict(ckpt["model_state"]); model.eval()

# ─── evaluation ───────────────────────────────────────
all_pred, all_true, all_prob = [], [], []
sigmoid = torch.nn.Sigmoid()

with torch.no_grad():
    for xb, yb in val_loader:
        logits = model(xb.to(device))          # (B,T,M)
        prob   = sigmoid(logits).cpu()
        pred   = (prob >= 0.5).int()
        all_prob.append(prob)
        all_pred.append(pred)
        all_true.append(yb.int())

prob = torch.cat(all_prob, 0)   # (N,T,M)
pred = torch.cat(all_pred, 0)
true = torch.cat(all_true, 0)

# strict match
strict_acc = (pred == true).all(dim=(1,2)).float().mean().item()

# flatten for micro metrics
prob_flat = prob.view(-1).numpy()
pred_flat = pred.view(-1).numpy()
true_flat = true.view(-1).numpy()

try:
    auroc_micro = roc_auc_score(true_flat, prob_flat)
except ValueError:
    auroc_micro = np.nan
auprc_micro = average_precision_score(true_flat, prob_flat)
f1_micro    = f1_score(true_flat, pred_flat, average="micro", zero_division=0)

# ─── output ───────────────────────────────────────────
print("\n==== Seq-to-Seq Validation (Strict) ====")
print(f"Strict Match Accuracy : {strict_acc:.4f}")
print(f"AUROC  (micro)       : {auroc_micro:.4f}")
print(f"AUPRC  (micro)       : {auprc_micro:.4f}")
print(f"F1@0.5 (micro)       : {f1_micro:.4f}")

# ─── Delay Time Evaluation ──────────────────────────────
delays = []  # 지연 시간 (초)

for i in range(true.shape[0]):  # 샘플 수만큼 반복
    for m in range(true.shape[2]):  # 각 모터 M
        gt_seq = true[i, :, m].numpy()     # (T,)
        pr_seq = pred[i, :, m].numpy()     # (T,)

        # 실제 고장 시점
        if 1 in gt_seq:
            t_true = np.argmax(gt_seq == 1)

            # 예측에서 고장 감지 시점
            if 1 in pr_seq:
                t_pred = np.argmax(pr_seq == 1)
                delay = max(t_pred - t_true, 0) * 0.01  # 초 단위
                delays.append(delay)
            else:
                # 예측을 못한 경우 → 최대 지연으로 간주 (전체 길이 - 고장 시점)
                delay = (len(gt_seq) - t_true) * 0.01
                delays.append(delay)

if delays:
    avg_delay = np.mean(delays)
    print(f"\n📊 Average Fault Detection Delay: {avg_delay:.4f} seconds")
    print(f"🔧 Detected {len(delays)} faulty motor sequences.")
else:
    print("\n⚠️ No fault events detected in ground-truth.")

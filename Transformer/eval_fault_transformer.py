import os, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from fault_diagnosis_model import FaultDiagnosisTransformer  
import matplotlib.pyplot as plt


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
# ─── Additional Fault Detection Metrics ───────────────────────────────

frame_hz = 100  # 0.01초 단위 프레임이면 100Hz
k_frames = int(0.05 * frame_hz)   # ±50ms 허용
early_N  = int(0.2 * frame_hz)   # 200ms 내 정밀도

shift_tolerant_tp = 0
shift_tolerant_fp = 0
shift_tolerant_fn = 0

motor_correct = 0
motor_total   = 0

ious = []
fn_count = 0
tp_count = 0

prec_early_tp = 0
prec_early_fp = 0

flip_changes = 0
flip_total   = 0

for i in range(true.shape[0]):      # 각 샘플
    for m in range(true.shape[2]):  # 각 모터
        gt_seq = true[i, :, m].numpy()
        pr_seq = pred[i, :, m].numpy()

        # --- 1) Shift-tolerant F1 계산 ---
        if 1 in gt_seq:
            t_true_onset = np.argmax(gt_seq == 1)

            # 예측 온셋 ±k 프레임 매칭
            onset_match = False
            if 1 in pr_seq:
                t_pred_onset = np.argmax(pr_seq == 1)
                if abs(t_pred_onset - t_true_onset) <= k_frames:
                    shift_tolerant_tp += 1
                    onset_match = True
                else:
                    shift_tolerant_fp += 1
            else:
                shift_tolerant_fn += 1

            # --- 2) Fault Localization Accuracy ---
            if onset_match:
                # 여기서는 GT에서 m번 모터만 1이면 맞춘 것으로 간주
                gt_fault_ids = np.where(gt_seq == 1)[0]
                if m in gt_fault_ids:
                    motor_correct += 1
                motor_total += 1

            # --- 3) Segment IoU ---
            gt_idx = np.where(gt_seq == 1)[0]
            pr_idx = np.where(pr_seq == 1)[0]
            if len(gt_idx) > 0 or len(pr_idx) > 0:
                inter = len(np.intersect1d(gt_idx, pr_idx))
                union = len(np.union1d(gt_idx, pr_idx))
                ious.append(inter / union if union > 0 else 0)

            # --- 4) FNR ---
            if 1 in gt_seq:
                if not (1 in pr_seq):
                    fn_count += 1
                else:
                    tp_count += 1

            # --- 5) Precision@early N ---
            if 1 in gt_seq:
                start = t_true_onset
                end   = min(t_true_onset + early_N, len(pr_seq))
                early_preds = pr_seq[start:end]
                if len(early_preds) > 0:
                    prec_early_tp += np.sum((early_preds == 1) & (gt_seq[start:end] == 1))
                    prec_early_fp += np.sum((early_preds == 1) & (gt_seq[start:end] == 0))

        # --- 6) Flip Rate ---
        flips = np.sum(pr_seq[1:] != pr_seq[:-1])
        flip_changes += flips
        flip_total   += len(pr_seq) - 1

# Shift-tolerant F1
if shift_tolerant_tp + shift_tolerant_fp + shift_tolerant_fn > 0:
    shift_f1 = (2 * shift_tolerant_tp) / (
        2 * shift_tolerant_tp + shift_tolerant_fp + shift_tolerant_fn
    )
else:
    shift_f1 = np.nan

# Fault Localization Accuracy
motor_acc = motor_correct / motor_total if motor_total > 0 else np.nan

# Segment IoU
mean_iou = np.mean(ious) if ious else np.nan

# FNR
fnr = fn_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else np.nan

# Precision@early N
prec_early = prec_early_tp / (prec_early_tp + prec_early_fp) if (prec_early_tp + prec_early_fp) > 0 else np.nan

# Flip Rate
flip_rate = flip_changes / flip_total if flip_total > 0 else np.nan

print("\n==== Additional Fault Detection Metrics ====")
print(f"Shift-tolerant F1 (±{k_frames} frames): {shift_f1:.4f}")
print(f"Fault Localization Accuracy           : {motor_acc:.4f}")
print(f"Segment IoU                            : {mean_iou:.4f}")
print(f"False Negative Rate                    : {fnr:.4f}")
print(f"Precision@early {early_N} frames       : {prec_early:.4f}")
print(f"Flip Rate                              : {flip_rate:.4f}")

# ─── Detection Delay Histogram (Percentage, 0~0.2s) ─────────────────────────────
if delays:
    delays_arr = np.array(delays)

    # 비율(%) 히스토그램 계산
    counts, bins = np.histogram(delays_arr, bins=50, range=(0, 0.2))
    percentages = (counts / len(delays_arr)) * 100

    # 그래프
    plt.figure(figsize=(6, 4))
    plt.bar(bins[:-1], percentages, width=(bins[1]-bins[0]),
            color='skyblue', edgecolor='black', align='edge')
    plt.axvline(0.05, color='green', linestyle='--', label='0.05s')
    plt.axvline(0.1,  color='orange', linestyle='--', label='0.1s')
    plt.axvline(0.2,  color='red', linestyle='--', label='0.2s')
    plt.xlabel('Detection Delay (seconds)')
    plt.ylabel('Percentage of Cases (%)')
    plt.title('Detection Delay Distribution (0~0.2s)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 파일 저장
    save_path_pct = os.path.join(repo_root, "delay_hist_percentage.png")
    plt.savefig(save_path_pct, dpi=300)
    print(f"\n📁 Percentage histogram saved to: {save_path_pct}")

    # 주요 구간 비율 출력
    within_005 = np.mean(delays_arr <= 0.05)
    within_010 = np.mean(delays_arr <= 0.1)
    within_020 = np.mean(delays_arr <= 0.2)
    print(f"⏱ Delay <= 0.05s : {within_005*100:.2f}% of cases")
    print(f"⏱ Delay <= 0.10s : {within_010*100:.2f}% of cases")
    print(f"⏱ Delay <= 0.20s : {within_020*100:.2f}% of cases")
else:
    print("\n⚠️ No fault events to plot delay distribution.")

# 탐지 성공 케이스만 평균
success_delays = []

for i in range(true.shape[0]):  # 샘플 수
    for m in range(true.shape[2]):  # 모터 수
        gt_seq = true[i, :, m].numpy()
        pr_seq = pred[i, :, m].numpy()

        if 1 in gt_seq:  # 실제 고장 있음
            t_true = np.argmax(gt_seq == 1)
            if 1 in pr_seq:  # 탐지 성공
                t_pred = np.argmax(pr_seq == 1)
                delay = max(t_pred - t_true, 0) * 0.01
                success_delays.append(delay)

# 출력
if success_delays:
    avg_success = np.mean(success_delays)
    median_success = np.median(success_delays)
    print(f"\n✅ Avg Delay (Detected only): {avg_success:.4f} s")
    print(f"   Median (Detected only)  : {median_success:.4f} s")
    print(f"   Cases counted           : {len(success_delays)}")
else:
    print("\n⚠️ No successful detections to calculate delay.")

# ───────────────────────────────────────────────────────────────
# 상위 비율 제거 평균 (내림차순)
for pct in [97, 97.5, 98, 98.5, 99, 99.5]:  # 상위 3%, 2.5%, 2%, 1.5%, 1%, 0.5%
    remove_ratio = 100 - pct
    cutoff = np.percentile(delays_arr, pct)
    filtered_delays = delays_arr[delays_arr <= cutoff]
    
    avg_trimmed = np.mean(filtered_delays)
    median_trimmed = np.median(filtered_delays)
    removed_cases = len(delays_arr) - len(filtered_delays)
    
    print(f"\n📊 Avg Delay (Top {remove_ratio:.1f}% removed): {avg_trimmed:.4f} s")
    print(f"   Median Delay  : {median_trimmed:.4f} s")
    print(f"   Cutoff value  : {cutoff:.4f} s")
    print(f"   Removed cases : {removed_cases} / {len(delays_arr)}")

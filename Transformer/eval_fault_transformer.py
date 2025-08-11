# Transformer/eval_fault_transformer.py
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from fault_diagnosis_model import FaultDiagnosisTransformer
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
seed       = 42
print("device:", device)

# ── Paths ──────────────────────────────────────────────
link_count = int(input("How many links?: ") or 1)
data_path  = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")
ckpt_path  = os.path.join(repo_root, "Transformer", f"Transformer_link_{link_count}.pth")

# ── Load data ─────────────────────────────────────────
data    = np.load(data_path)
desired = data["desired"]                      # (S,T,4,4)
actual  = data["actual"]
labels  = data["label"].astype(np.float32)     # (S,T,M)  # 1=정상, 0=고장
dt      = float(data.get("dt", 0.01))
frame_hz= 1.0 / dt

S, T, _, _ = desired.shape
M = labels.shape[2]

des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)
X  = np.concatenate([des_12, act_12], axis=2).astype(np.float32)   # (S,T,24)
y  = labels                                                         # (S,T,M)

# ── Load checkpoint ────────────────────────────────────
try:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
except Exception:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

mean, std  = ckpt["train_mean"], ckpt["train_std"]
if isinstance(mean, torch.Tensor): mean = mean.cpu().numpy()
if isinstance(std, torch.Tensor):  std  = std.cpu().numpy()
cfg        = ckpt["cfg"]
assert (ckpt["input_dim"], ckpt["T"], ckpt["M"]) == (24, T, M), "shape mismatch"

# ── Normalize & val split ─────────────────────────────
X = (X - mean) / std
ds_all = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
train_sz = int(0.8 * S); val_sz = S - train_sz
_, val_ds = random_split(ds_all, [train_sz, val_sz],
                         generator=torch.Generator().manual_seed(seed))
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# ── Rebuild model ─────────────────────────────────────
model = FaultDiagnosisTransformer(
    input_dim=24,
    d_model=cfg["d_model"], nhead=cfg["nhead"],
    num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
    dropout=cfg["dropout"], output_dim=M, max_seq_len=T
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── Inference ─────────────────────────────────────────
all_prob, all_pred, all_true = [], [], []
sigmoid = torch.nn.Sigmoid()
with torch.no_grad():
    for xb, yb in val_loader:
        logits = model(xb.to(device))          # (B,T,M)
        prob   = sigmoid(logits).cpu()         # 예측확률(p=정상일 확률)
        pred   = (prob >= 0.5).int()           # 1=정상, 0=고장
        all_prob.append(prob)
        all_pred.append(pred)
        all_true.append(yb.int())

prob = torch.cat(all_prob, 0)   # (N,T,M)  p(normal)
pred = torch.cat(all_pred, 0)   # (N,T,M)  1=normal,0=fault
true = torch.cat(all_true, 0)   # (N,T,M)  1=normal,0=fault
N = true.shape[0]

# ======================== 유틸 =========================
def first_zero_idx(seq_0or1: np.ndarray, min_run: int = 1):
    """값==0(고장)이 min_run 프레임 이상 연속으로 처음 등장하는 시점. 없으면 None."""
    if min_run <= 1:
        return int(np.argmax(seq_0or1 == 0)) if (seq_0or1 == 0).any() else None
    run = 0
    for t, v in enumerate(seq_0or1):
        run = run + 1 if v == 0 else 0
        if run >= min_run:
            return t - min_run + 1
    return None

def run_filter_zero_mask(seq_0or1: np.ndarray, min_run: int = 1) -> np.ndarray:
    """0(고장)이 연속 min_run 이상인 구간만 True로 인정하는 마스크."""
    if min_run <= 1:
        return (seq_0or1 == 0)
    T = len(seq_0or1)
    mask = np.zeros(T, dtype=bool)
    run = 0
    for t, v in enumerate(seq_0or1):
        run = run + 1 if v == 0 else 0
        if run >= min_run:
            mask[t - min_run + 1 : t + 1] = True
    return mask

def fault_set(mat_TxM: np.ndarray, min_run: int = 1):
    """(T,M)에서 0(고장)인 모터 ID 집합 반환."""
    T, M = mat_TxM.shape
    s = set()
    for m in range(M):
        if first_zero_idx(mat_TxM[:, m], min_run=min_run) is not None:
            s.add(m)
    return s

# ============ Top-K 패턴 매칭(다수 패턴) 유틸 =============
def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

def top_k_patterns_with_tol(mat_TxM: np.ndarray, k=2, tol=0):
    """
    행 패턴을 tol 해밍 반경으로 간이 클러스터링하여 카운트.
    반환: (patterns[list[np.ndarray]], counts[list[int]], coverage[0..1])
    """
    T, M = mat_TxM.shape
    protos = []   # 리스트[(패턴(np.array), count)]
    for t in range(T):
        row = mat_TxM[t]
        assigned = False
        for idx, (p, c) in enumerate(protos):
            if hamming(row, p) <= tol:
                protos[idx] = (p, c+1)
                assigned = True
                break
        if not assigned:
            protos.append((row.copy(), 1))
    protos_sorted = sorted(protos, key=lambda pc: pc[1], reverse=True)
    top = protos_sorted[:k]
    total = float(T)
    patterns = [p for p, _ in top]
    counts   = [c for _, c in top]
    coverage = sum(counts) / total if total > 0 else 0.0
    return patterns, counts, coverage

def equal_pattern_multiset(pats_a, pats_b):
    """top-k 패턴들을 멀티셋 비교(순서 무시, 중복 허용)."""
    if len(pats_a) != len(pats_b):
        return False
    aa = [tuple(x.tolist()) for x in pats_a]
    bb = [tuple(x.tolist()) for x in pats_b]
    from collections import Counter
    return Counter(aa) == Counter(bb)

def first_index_close(mat_TxM: np.ndarray, pattern: np.ndarray, tol: int = 0):
    """해밍 거리 tol 이내로 일치하는 첫 프레임 인덱스 (없으면 None)."""
    for t in range(mat_TxM.shape[0]):
        if hamming(mat_TxM[t], pattern) <= tol:
            return t
    return None

# ======================================================
# A) 마이크로 지표(AUROC/AUPRC/F1) — 양성=고장으로 계산
prob_fault = (1.0 - prob).view(-1).numpy()          # 양성=고장 확률
true_fault = (1 - true).view(-1).numpy()            # 1=고장, 0=정상
pred_fault = (1 - pred).view(-1).numpy()            # 1=고장, 0=정상

try:
    auroc_micro = roc_auc_score(true_fault, prob_fault)
except ValueError:
    auroc_micro = np.nan
auprc_micro = average_precision_score(true_fault, prob_fault)
f1_micro    = f1_score(true_fault, pred_fault, average="micro", zero_division=0)

print("\n==== Micro metrics (positive = FAULT) ====")
print(f"AUROC  : {auroc_micro:.4f}")
print(f"AUPRC  : {auprc_micro:.4f}")
print(f"F1@0.5 : {f1_micro:.4f}")

# ======================================================
# B) 이벤트 지표(모터 단위 TP/FP/FN) + Delay(성공 탐지만)
tp = fp = fn = 0
detected_delays = []  # seconds
for i in range(N):
    gt = true[i].numpy()   # 1=정상,0=고장
    pr = pred[i].numpy()
    for m in range(M):
        gt_seq = gt[:, m]; pr_seq = pr[:, m]
        if 0 in gt_seq:                      # 실제 고장
            t_true = int(np.argmax(gt_seq == 0))
            if 0 in pr_seq:                  # 해당 모터 탐지
                t_pred = int(np.argmax(pr_seq == 0))
                tp += 1
                detected_delays.append(max(t_pred - t_true, 0) * dt)
            else:
                fn += 1
        else:
            if 0 in pr_seq:
                fp += 1

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_event  = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

print("\n==== [Event-level] motor-wise detection ====")
print(f"TP events: {tp} / {tp + fn}")
print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1_event:.4f}")

if detected_delays:
    d = np.array(detected_delays)
    print(f"Delay (TP only) → mean={d.mean():.4f}s, median={np.median(d):.4f}s, n={len(d)}")
else:
    print("Delay: (no TP)")

# ======================================================
# C) Strict 샘플 정확도 (모터 ID 집합 완전 일치, 시간 무시)
MIN_RUN_PRED = 1  # 예측 스파이크 억제 원하면 2~3으로
strict_ok = 0
for i in range(N):
    gt = true[i].numpy(); pr = pred[i].numpy()
    gt_set = fault_set(gt, min_run=1)
    pr_set = fault_set(pr, min_run=MIN_RUN_PRED)
    if gt_set == pr_set:
        strict_ok += 1
print("\n==== [Strict] sample accuracy (ID set equality) ====")
print(f"Strict acc: {strict_ok}/{N} = {strict_ok/max(N,1):.4f}")

# ======================================================
# D) Lenient 샘플 정확도 (고장 구간 90% 이상 맞추면 성공) + Delay(성공만)
TAU_RECALL   = 0.90   # 90% 규칙
MAX_FP_RATE  = 0.10   # 정상 프레임 FP율 제한(끄려면 None)
MIN_RUN_PRED = 1      # 예측 스파이크 억제

lenient_ok = 0
lenient_delays = []

for i in range(N):
    gt = true[i].numpy(); pr = pred[i].numpy()
    gt_fault_motors = [m for m in range(M) if (gt[:, m] == 0).any()]
    if not gt_fault_motors:
        continue

    # 모터별 시간 기반 recall
    recalls = []
    per_motor_delays = []
    for m in gt_fault_motors:
        gt_fault_mask = (gt[:, m] == 0)
        pr_fault_mask = run_filter_zero_mask(pr[:, m], MIN_RUN_PRED)
        denom = gt_fault_mask.sum()
        if denom == 0:
            continue
        recalls.append(((gt_fault_mask & pr_fault_mask).sum()) / denom)

        t_true = first_zero_idx(gt[:, m], 1)
        t_pred = first_zero_idx(pr[:, m], MIN_RUN_PRED)
        if t_true is not None and t_pred is not None:
            per_motor_delays.append(max(t_pred - t_true, 0) * dt)

    if not recalls:
        continue
    mean_rec = float(np.mean(recalls))

    fp_ok = True
    if MAX_FP_RATE is not None:
        gt_normal_mask = (gt == 1)
        pr_fault_mask_all = np.stack(
            [run_filter_zero_mask(pr[:, m], MIN_RUN_PRED) for m in range(M)], axis=1
        )
        fp_num = (pr_fault_mask_all & gt_normal_mask).sum()
        fp_den = gt_normal_mask.sum()
        fp_rate = fp_num / max(fp_den, 1)
        fp_ok = (fp_rate <= MAX_FP_RATE)

    if (mean_rec >= TAU_RECALL) and fp_ok:
        lenient_ok += 1
        lenient_delays.extend(per_motor_delays)

print("\n==== [Lenient] sample accuracy (≥90% time coverage) ====")
print(f"Lenient acc: {lenient_ok}/{N} = {lenient_ok/max(N,1):.4f}")
if lenient_delays:
    d = np.array(lenient_delays)
    print(f"Lenient delay (success only) → mean={d.mean():.4f}s, median={np.median(d):.4f}s, n={len(lenient_delays)}")
else:
    print("Lenient delay: (no lenient successes)")

# ======================================================
# E) Delay 분포 히스토그램 (TP 기준) ─ 0~0.2s 구간 비율
if detected_delays:
    d = np.array(detected_delays)
    counts, bins = np.histogram(d, bins=50, range=(0.0, 0.2))
    pct = counts / len(d) * 100.0

    plt.figure(figsize=(6,4))
    plt.bar(bins[:-1], pct, width=(bins[1]-bins[0]), edgecolor='black', align='edge')
    plt.axvline(0.05, linestyle='--', label='0.05s')
    plt.axvline(0.10, linestyle='--', label='0.10s')
    plt.axvline(0.20, linestyle='--', label='0.20s')
    plt.xlabel('Detection Delay (s)'); plt.ylabel('Percentage of Cases (%)')
    plt.title('Detection Delay Distribution (TP only, 0–0.2s)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    save_path_pct = os.path.join(repo_root, "delay_hist_percentage.png")
    plt.savefig(save_path_pct, dpi=300)
    print(f"\n📁 Percentage histogram saved to: {save_path_pct}")

    within_005 = np.mean(d <= 0.05); within_010 = np.mean(d <= 0.10); within_020 = np.mean(d <= 0.20)
    print(f"⏱ Delay ≤ 0.05s : {within_005*100:.2f}%")
    print(f"⏱ Delay ≤ 0.10s : {within_010*100:.2f}%")
    print(f"⏱ Delay ≤ 0.20s : {within_020*100:.2f}%")
else:
    print("\n(No TP delays to plot)")

# ======================================================
# F) Majority Top-2 패턴 매칭 정확도 + Delay(성공만)
TOP_K = 2
HAMMING_TOL = 0     # 패턴 허용 해밍 거리(작은 오류 흡수). 0~2 권장
COVERAGE_TAU = 0.8  # top-2 패턴이 전체 프레임의 80% 이상을 덮어야 유효

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

def top_k_patterns_with_tol(mat_TxM: np.ndarray, k=2, tol=0):
    """
    행 패턴을 tol 해밍 반경으로 간이 클러스터링하여 카운트.
    반환: (patterns[list[np.ndarray]], counts[list[int]], coverage[0..1])
    """
    T, M = mat_TxM.shape
    protos = []   # 리스트[(패턴(np.array), count)]
    for t in range(T):
        row = mat_TxM[t]
        assigned = False
        for idx, (p, c) in enumerate(protos):
            if hamming(row, p) <= tol:
                protos[idx] = (p, c+1)
                assigned = True
                break
        if not assigned:
            protos.append((row.copy(), 1))
    protos_sorted = sorted(protos, key=lambda pc: pc[1], reverse=True)
    top = protos_sorted[:k]
    total = float(T)
    patterns = [p for p, _ in top]
    counts   = [c for _, c in top]
    coverage = (sum(counts) / total) if total > 0 else 0.0
    return patterns, counts, coverage

def can_match_topk(pats_a, pats_b, tol=0):
    """
    pats_a, pats_b: list[np.ndarray] (길이 동일)
    해밍 거리 ≤ tol 조건으로 1:1 매칭이 모두 성립하면 True.
    (순서 무시, 카운트는 이미 top-k에 반영되었다고 보고 패턴 내용만 매칭)
    """
    if len(pats_a) != len(pats_b):
        return False
    used = [False] * len(pats_b)
    for a in pats_a:
        found = False
        for j, b in enumerate(pats_b):
            if not used[j] and hamming(a, b) <= tol:
                used[j] = True
                found = True
                break
        if not found:
            return False
    return True

def first_index_close(mat_TxM: np.ndarray, pattern: np.ndarray, tol: int = 0):
    """해밍 거리 tol 이내로 일치하는 첫 프레임 인덱스 (없으면 None)."""
    for t in range(mat_TxM.shape[0]):
        if hamming(mat_TxM[t], pattern) <= tol:
            return t
    return None

maj_ok = 0
maj_delays = []  # seconds

for i in range(N):
    gt = true[i].numpy()   # (T,M)
    pr = pred[i].numpy()

    gt_pats, gt_cnts, gt_cov = top_k_patterns_with_tol(gt, k=TOP_K, tol=HAMMING_TOL)
    pr_pats, pr_cnts, pr_cov = top_k_patterns_with_tol(pr, k=TOP_K, tol=HAMMING_TOL)

    # 커버리지 낮으면 노이즈가 많다고 보고 실패 처리(원하면 완화 가능)
    if (gt_cov < COVERAGE_TAU) or (pr_cov < COVERAGE_TAU):
        continue

    # 최다 top-k 패턴이 tol 이내에서 1:1 매칭되면 정답
    if can_match_topk(gt_pats, pr_pats, tol=HAMMING_TOL):
        maj_ok += 1

        # Delay: "고장 패턴"(행에 0이 하나라도 있는 패턴)을 찾아 첫 등장 시점 차이
        def pick_fault_pattern(pats, cnts):
            idxs = [idx for idx, p in enumerate(pats) if (p == 0).any()]
            if not idxs:
                return None, None
            # 고장 패턴 중에서 카운트가 가장 큰 것을 대표로 선택
            best = max(idxs, key=lambda j: cnts[j])
            return pats[best], cnts[best]

        gt_fault_pat, _ = pick_fault_pattern(gt_pats, gt_cnts)
        pr_fault_pat, _ = pick_fault_pattern(pr_pats, pr_cnts)

        if gt_fault_pat is not None and pr_fault_pat is not None:
            t_true = first_index_close(gt, gt_fault_pat, tol=HAMMING_TOL)
            t_pred = first_index_close(pr, pr_fault_pat, tol=HAMMING_TOL)
            if (t_true is not None) and (t_pred is not None):
                maj_delays.append(max(t_pred - t_true, 0) * dt)

print("\n==== [Majority Top-2] pattern-match accuracy ====")
print(f"Majority top-2 acc: {maj_ok}/{N} = {maj_ok/max(N,1):.4f}")
if maj_delays:
    d = np.array(maj_delays)
    print(f"Majority delay (success only) → mean={d.mean():.4f}s, median={np.median(d):.4f}s, n={len(maj_delays)}")
else:
    print("Majority delay: (no majority successes)")

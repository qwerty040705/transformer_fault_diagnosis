# Transformer/eval_fault_transformer.py
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# ── Config ─────────────────────────────────────────────
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
seed       = 42
print("device:", device)

# ====== 지연/후처리 하이퍼파라미터 ======
MAX_LAG_SEC   = 0.05  # shift-tolerant exact-all: ±이만큼 시프트 허용(초)
PRE_TOL_SEC   = 0.00  # 온셋 탐지 허용 윈도우(초) - 앞쪽(조기경보 허용폭)
POST_TOL_SEC  = 0.05  # 온셋 탐지 허용 윈도우(초) - 뒤쪽(지연 허용폭)
MIN_RUN_PRED  = 1     # 예측 스파이크 억제(연속 0 프레임 최소길이). 1=끄기, 2~3 권장

# ====== Top-K 패턴 매칭 설정 ======
TOP_K         = 2     # Majority Top-2
HAMMING_TOL   = 0     # 패턴 일치 허용 해밍 거리(0~2 권장)
COVERAGE_TAU  = 0.8   # 상위 K 패턴의 최소 커버리지

# ── Paths ──────────────────────────────────────────────
try:
    link_count = int(input("How many links?: ").strip())
except Exception:
    link_count = 1
    print("[WARN] Invalid input. Fallback to link_count=1")
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
cfg        = dict(ckpt.get("cfg", {}))  # 호환성
cfg.setdefault("d_model", 64)
cfg.setdefault("nhead", 8)
cfg.setdefault("num_layers", 2)
cfg.setdefault("dim_feedforward", 128)
cfg.setdefault("dropout", 0.1)
cfg.setdefault("posenc", "learned")   # "learned" | "sincos" | "rope"
cfg.setdefault("rope_base", 10000.0)  # RoPE base

assert (ckpt["input_dim"], ckpt["T"], ckpt["M"]) == (24, T, M), "shape mismatch with checkpoint"

# ── Normalize & val split ─────────────────────────────
X = (X - mean) / std
torch.manual_seed(seed); np.random.seed(seed)
ds_all = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
train_sz = int(0.8 * S); val_sz = S - train_sz
_, val_ds = random_split(ds_all, [train_sz, val_sz],
                         generator=torch.Generator().manual_seed(seed))
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# ── Model (with posenc option incl. RoPE) ─────────────────────────────
def build_model(input_dim, output_dim, max_seq_len, cfg):
    posenc = cfg.get("posenc", "learned")
    rope_base = cfg.get("rope_base", 10000.0)

    # ---- RoPE 유틸 ----
    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
            super().__init__()
            inv_freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)  # (dim/2,)
            t = torch.arange(max_seq_len, dtype=torch.float32)  # (T,)
            freqs = torch.outer(t, inv_freq)  # (T, dim/2)
            cos = torch.zeros(max_seq_len, dim)
            sin = torch.zeros(max_seq_len, dim)
            cos[:, 0::2] = torch.cos(freqs)
            sin[:, 0::2] = torch.sin(freqs)
            cos[:, 1::2] = torch.cos(freqs)
            sin[:, 1::2] = torch.sin(freqs)
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)
        def get_cos_sin(self, T: int):
            return self.cos_cached[:T], self.sin_cached[:T]

    def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
        sin = sin.unsqueeze(0).unsqueeze(0)
        def rotate(x):
            x_even = x[..., 0::2]; x_odd  = x[..., 1::2]
            x_rot_even = x_even * cos[..., 0::2] - x_odd * sin[..., 0::2]
            x_rot_odd  = x_odd  * cos[..., 1::2] + x_even * sin[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = x_rot_even; out[..., 1::2] = x_rot_odd
            return out
        return rotate(q), rotate(k)

    class RotaryEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="relu", norm_first=True,
                     max_seq_len=1024, base=10000.0):
            super().__init__()
            assert d_model % nhead == 0
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm_first = norm_first
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.act = nn.ReLU() if activation == "relu" else nn.GELU()
            self.nhead = nhead
            self.head_dim = d_model // nhead
            self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=base)
        def _mha_block(self, x):
            B, T, D = x.size(); H = self.nhead; Hd = self.head_dim
            W = self.self_attn.in_proj_weight; b = self.self_attn.in_proj_bias
            q_proj = torch.addmm(b[:D],     x.view(-1, D), W[:D, :].t()).view(B, -1, D)
            k_proj = torch.addmm(b[D:2*D],  x.view(-1, D), W[D:2*D, :].t()).view(B, -1, D)
            v_proj = torch.addmm(b[2*D:],   x.view(-1, D), W[2*D:, :].t()).view(B, -1, D)
            def shape(xp): return xp.view(B, -1, H, Hd).transpose(1, 2)
            q = shape(q_proj); k = shape(k_proj); v = shape(v_proj)
            cos, sin = self.rotary.get_cos_sin(T)
            q_rot, k_rot = apply_rotary(q, k, cos, sin)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q_rot, k_rot, v, dropout_p=self.self_attn.dropout.p if self.training else 0.0
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
            out = self.self_attn.out_proj(attn_output)
            return out
        def _ff_block(self, x):
            return self.linear2(self.dropout(self.act(self.linear1(x))))
        def forward(self, src: torch.Tensor):
            if self.norm_first:
                x = src + self.dropout1(self._mha_block(self.norm1(src)))
                x = x   + self.dropout2(self._ff_block(self.norm2(x)))
            else:
                x = self.norm1(src + self.dropout1(self._mha_block(src)))
                x = self.norm2(x   + self.dropout2(self._ff_block(x)))
            return x

    class FaultDiagnosisTransformer(nn.Module):
        def __init__(self, input_dim=24, d_model=64, nhead=8, num_layers=2,
                     dim_feedforward=128, dropout=0.1, output_dim=8,
                     max_seq_len=1000, posenc="learned", rope_base=10000.0):
            super().__init__()
            self.max_seq_len = max_seq_len
            self.d_model = d_model
            self.posenc = posenc
            self.input_proj = nn.Linear(input_dim, d_model)
            if posenc == "learned":
                self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
                nn.init.normal_(self.pos_embedding, std=0.02)
                self.use_additive_pos = True
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, activation="relu", batch_first=True, norm_first=True
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
            elif posenc == "sincos":
                pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
                position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer("pos_embedding", pe, persistent=False)
                self.use_additive_pos = True
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, activation="relu", batch_first=True, norm_first=True
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
            elif posenc == "rope":
                self.use_additive_pos = False
                layers = []
                for _ in range(num_layers):
                    layers.append(RotaryEncoderLayer(
                        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                        dropout=dropout, activation="relu", norm_first=True,
                        max_seq_len=max_seq_len, base=rope_base
                    ))
                self.encoder = nn.Sequential(*layers)
            else:
                raise ValueError(f"Unknown posenc: {posenc}")
            self.pos_drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim),
            )
            nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, _ = x.shape
            if T > self.max_seq_len:
                raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}.")
            z = self.input_proj(x) / math.sqrt(self.d_model)
            if getattr(self, "use_additive_pos", False):
                z = z + self.pos_embedding[:, :T, :]
            z = self.pos_drop(z)
            z = self.encoder(z)
            return self.head(z)

    return FaultDiagnosisTransformer(
        input_dim=input_dim,
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"], output_dim=output_dim, max_seq_len=max_seq_len,
        posenc=posenc, rope_base=rope_base,
    ).to(device)

model = build_model(24, M, T, cfg)
strict_load = True
if cfg.get("posenc", "learned") != "learned":
    strict_load = False
missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=strict_load)
if not strict_load:
    print(f"[INFO] Loaded with strict=False. missing={len(missing)}, unexpected={len(unexpected)}")
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

prob = torch.cat(all_prob, 0)   # (N,T,M)
pred = torch.cat(all_pred, 0)   # (N,T,M)
true = torch.cat(all_true, 0)   # (N,T,M)
true_np = true.numpy()
pred_np = pred.numpy()
N, T, M = true_np.shape

# ── (선택) 예측 스파이크 억제 ─────────────────────────
def run_filter_zero_mask(seq_0or1: np.ndarray, min_run: int = 1) -> np.ndarray:
    if min_run <= 1:
        return (seq_0or1 == 0)
    Tlen = len(seq_0or1)
    mask = np.zeros(Tlen, dtype=bool)
    run = 0
    for t, v in enumerate(seq_0or1):
        run = run + 1 if v == 0 else 0
        if run >= min_run:
            mask[t - min_run + 1 : t + 1] = True
    return mask

if MIN_RUN_PRED > 1:
    pred_filt = pred_np.copy()
    for i in range(N):
        for m in range(M):
            mask_fault = run_filter_zero_mask(pred_filt[i, :, m], min_run=MIN_RUN_PRED)
            pred_filt[i, :, m] = np.where(mask_fault, 0, 1)
    pred_np = pred_filt

# ===================== Frame-wise ALL-motors metrics ======================
eq_frame = (true_np == pred_np).all(axis=2)   # (N,T)
overall_frame_acc      = eq_frame.mean()
per_sample_frame_acc   = eq_frame.mean(axis=1)

print("\n==== [Exact-all motors] Frame-wise accuracy ====")
print(f"Overall frame accuracy (exact-all) : {overall_frame_acc:.4f}")
print(f"Per-sample mean accuracy           : {per_sample_frame_acc.mean():.4f}")
print(f"Per-sample median accuracy         : {np.median(per_sample_frame_acc):.4f}")

is_fault_frame = (true_np == 0).any(axis=2)
is_norm_frame  = ~is_fault_frame
acc_on_fault   = (eq_frame & is_fault_frame).sum() / max(is_fault_frame.sum(), 1)
acc_on_norm    = (eq_frame & is_norm_frame ).sum() / max(is_norm_frame.sum(), 1)
print(f"Exact-all on FAULT frames          : {acc_on_fault:.4f}")
print(f"Exact-all on NORMAL frames         : {acc_on_norm:.4f}")

# ================= Shift-tolerant exact-all (지연 관용) ===================
def exact_all_with_shift(gt_TxM: np.ndarray, pr_TxM: np.ndarray, max_lag: int):
    T_, M_ = gt_TxM.shape
    best_acc, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            pr_slice = pr_TxM[-lag:, :]
            gt_slice = gt_TxM[:T_+lag, :]
        elif lag > 0:
            pr_slice = pr_TxM[:T_-lag, :]
            gt_slice = gt_TxM[lag:, :]
        else:
            pr_slice = pr_TxM; gt_slice = gt_TxM
        if pr_slice.size == 0:
            continue
        eq = (gt_slice == pr_slice).all(axis=1)
        acc = eq.mean()
        if acc > best_acc:
            best_acc, best_lag = acc, lag
    return best_acc, best_lag

MAX_LAG = max(0, int(round(MAX_LAG_SEC / dt)))
best_accs, best_lags = [], []
for i in range(N):
    acc_i, lag_i = exact_all_with_shift(true_np[i], pred_np[i], MAX_LAG)
    best_accs.append(acc_i); best_lags.append(lag_i)

print("\n==== [Shift-tolerant] Exact-all motors (per frame) ====")
print(f"Best-acc (mean over samples)       : {np.mean(best_accs):.4f}")
print(f"Best-acc (median over samples)     : {np.median(best_accs):.4f}")
if MAX_LAG > 0:
    lags_sec = np.array(best_lags) * dt
    print(f"Best lag (sec) mean/median         : {lags_sec.mean():.4f} / {np.median(lags_sec):.4f}")
    pct_zero = np.mean(np.array(best_lags) == 0) * 100.0
    print(f"Best lag == 0 frame (%)            : {pct_zero:.2f}%")

# ======================= Sequence onset (전체 모터 기준) ===================
def first_fault_onset(mat_TxM: np.ndarray):
    fault_any = (mat_TxM == 0).any(axis=1)  # (T,)
    return int(np.argmax(fault_any)) if fault_any.any() else None

PRE_TOL  = int(round(PRE_TOL_SEC  / dt))
POST_TOL = int(round(POST_TOL_SEC / dt))

tp = fp = fn = 0
delays = []  # seconds (TP만)
for i in range(N):
    t_true = first_fault_onset(true_np[i])   # None이면 완전 정상 시퀀스
    t_pred = first_fault_onset(pred_np[i])
    if t_true is None:
        if t_pred is not None:
            fp += 1
    else:
        if t_pred is None:
            fn += 1
        else:
            if (t_pred >= t_true - PRE_TOL) and (t_pred <= t_true + POST_TOL):
                tp += 1
                delays.append(max(t_pred - t_true, 0) * dt)
            else:
                fn += 1

prec = tp / (tp + fp) if (tp + fp) else 0.0
rec  = tp / (tp + fn) if (tp + fn) else 0.0
f1_onset = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

print("\n==== [Onset (any-motor)] Sequence-level detection with tolerance ====")
print(f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1_onset:.4f}")
if delays:
    d = np.array(delays)
    print(f"Onset delay (TP only) mean/median  : {d.mean():.4f}s / {np.median(d):.4f}s (n={len(d)})")

# ======================= Majority Top-2 패턴 매칭 =========================
def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

def top_k_patterns_with_tol(mat_TxM: np.ndarray, k=2, tol=0):
    Tlen, Mdim = mat_TxM.shape
    protos = []  # (pattern, count)
    for t in range(Tlen):
        row = mat_TxM[t]
        assigned = False
        for idx, (p, c) in enumerate(protos):
            if hamming(row, p) <= tol:
                protos[idx] = (p, c + 1); assigned = True; break
        if not assigned:
            protos.append((row.copy(), 1))
    protos_sorted = sorted(protos, key=lambda pc: pc[1], reverse=True)
    top = protos_sorted[:k]
    total = float(Tlen)
    patterns = [p for p, _ in top]
    counts   = [c for _, c in top]
    coverage = (sum(counts) / total) if total > 0 else 0.0
    return patterns, counts, coverage

def can_match_topk(pats_a, pats_b, tol=0):
    if len(pats_a) != len(pats_b):
        return False
    used = [False] * len(pats_b)
    for a in pats_a:
        found = False
        for j, b in enumerate(pats_b):
            if not used[j] and hamming(a, b) <= tol:
                used[j] = True; found = True; break
        if not found:
            return False
    return True

def first_index_close(mat_TxM: np.ndarray, pattern: np.ndarray, tol: int = 0):
    for t in range(mat_TxM.shape[0]):
        if hamming(mat_TxM[t], pattern) <= tol:
            return t
    return None

maj_ok = 0
maj_delays = []
for i in range(N):
    gt = true_np[i]; pr = pred_np[i]
    gt_pats, gt_cnts, gt_cov = top_k_patterns_with_tol(gt, k=TOP_K, tol=HAMMING_TOL)
    pr_pats, pr_cnts, pr_cov = top_k_patterns_with_tol(pr, k=TOP_K, tol=HAMMING_TOL)
    if (gt_cov < COVERAGE_TAU) or (pr_cov < COVERAGE_TAU):
        continue
    if can_match_topk(gt_pats, pr_pats, tol=HAMMING_TOL):
        maj_ok += 1
        def pick_fault_pattern(pats, cnts):
            idxs = [idx for idx, p in enumerate(pats) if (p == 0).any()]
            if not idxs: return None, None
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

# ========================================================================
# ==================== 여기부터 '추가 지표' (비중복) =======================
# ========================================================================
# scikit-learn 의존 지표는 선택 사용
try:
    from sklearn.metrics import average_precision_score, f1_score, brier_score_loss
except Exception:
    average_precision_score = None
    f1_score = None
    brier_score_loss = None
    print("[WARN] scikit-learn이 없어 일부 지표(AUPRC/F1/Brier/ECE)가 비활성화됩니다.")

# ---- 헬퍼 (이름 충돌 방지용) ----
def _first_fault_onset_TxM(mat_TxM: np.ndarray):
    fault_any = (mat_TxM == 0).any(axis=1)
    return int(np.argmax(fault_any)) if fault_any.any() else None
def _first_fault_onset_1d(vec_T: np.ndarray):
    return int(np.argmax(vec_T == 0)) if (vec_T == 0).any() else None
def _segment_iou_from_onset(t_true, t_pred, Tlen):
    if t_true is None and t_pred is None: return 1.0
    if (t_true is None) ^ (t_pred is None): return 0.0
    a0, a1 = t_true, Tlen - 1
    b0, b1 = t_pred, Tlen - 1
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0
def _expected_calibration_error(y_true01, y_score, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum, total = 0.0, len(y_score)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        sel = (y_score >= lo) & (y_score < hi) if b < n_bins - 1 else (y_score >= lo) & (y_score <= hi)
        if sel.sum() == 0: continue
        conf = y_score[sel].mean(); acc = y_true01[sel].mean()
        ece_sum += (sel.sum() / total) * abs(acc - conf)
    return ece_sum

# (A) Any-fault Frame Accuracy  ← Exact-all과 비중복
any_fault_frame_acc = np.zeros((N, T), dtype=np.float32)
for i in range(N):
    for t in range(T):
        gt_fault_idx = np.where(true_np[i, t] == 0)[0]
        pr_fault_idx = np.where(pred_np[i, t] == 0)[0]
        if len(gt_fault_idx) == 0:
            any_fault_frame_acc[i, t] = 1.0 if len(pr_fault_idx) == 0 else 0.0
        else:
            any_fault_frame_acc[i, t] = 1.0 if len(np.intersect1d(gt_fault_idx, pr_fault_idx)) > 0 else 0.0

print("\n==== [Extra] Any-fault frame accuracy ====")
print(f"Any-fault (overall)               : {any_fault_frame_acc.mean():.4f}")

# (B) False Alarm Rate (FAR) / hour  ← 정상 시퀀스에서의 오경보
normal_idx = np.where([_first_fault_onset_TxM(true_np[i]) is None for i in range(N)])[0]
fa_count, normal_total_hours = 0, 0.0
for i in normal_idx:
    t_pred = _first_fault_onset_TxM(pred_np[i])
    if t_pred is not None:
        fa_count += 1
    normal_total_hours += (T * dt) / 3600.0
fa_per_hour = (fa_count / normal_total_hours) if normal_total_hours > 0 else 0.0

print("\n==== [Extra] False Alarm ====")
print(f"FA/hour on normal sequences       : {fa_per_hour:.6f}  (FA={fa_count}, hours={normal_total_hours:.3f})")

# (C) Time-To-Stable (TTS) after onset (any-motor, K=3)
K_STABLE = 3
tts_list_sec = []
for i in range(N):
    t_true = _first_fault_onset_TxM(true_np[i])
    if t_true is None: continue
    pred_any_fault = (pred_np[i] == 0).any(axis=1).astype(np.int32)
    run, t_stable = 0, None
    for t in range(t_true, T):
        if pred_any_fault[t] == 1:
            run += 1
            if run >= K_STABLE:
                t_stable = t; break
        else:
            run = 0
    if t_stable is not None:
        tts_list_sec.append((t_stable - t_true) * dt)

print("\n==== [Extra] Time-To-Stable (any-motor, K=3) ====")
if tts_list_sec:
    arr = np.array(tts_list_sec)
    print(f"TTS mean/median (sec)             : {arr.mean():.4f} / {np.median(arr):.4f} (n={len(arr)})")
else:
    print("TTS                                : (no measurable TTS)")

# (D) 모터별 온셋 P/R/F1 & Delay
print("\n==== [Extra] Per-motor Onset P/R/F1 & Delay ====")
for m in range(M):
    tp=fp=fn=0; delays_m=[]
    for i in range(N):
        t_true = _first_fault_onset_1d(true_np[i, :, m])
        t_pred = _first_fault_onset_1d(pred_np[i, :, m])
        if t_true is None:
            if t_pred is not None: fp += 1
        else:
            if t_pred is None:
                fn += 1
            else:
                if (t_pred >= t_true - PRE_TOL) and (t_pred <= t_true + POST_TOL):
                    tp += 1; delays_m.append(max(t_pred - t_true, 0) * dt)
                else:
                    fn += 1
    P = tp/(tp+fp) if (tp+fp) else 0.0
    R = tp/(tp+fn) if (tp+fn) else 0.0
    F1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
    D_mean = float(np.mean(delays_m)) if delays_m else None
    D_med  = float(np.median(delays_m)) if delays_m else None
    print(f"[m={m}] P/R/F1={P:.3f}/{R:.3f}/{F1:.3f}  Delay mean/median(s)={D_mean if D_mean is not None else 'NA'}/{D_med if D_med is not None else 'NA'}")

# (E) 모터별 프레임 AUPRC & F1 (fault=positive)
print("\n==== [Extra] Per-motor Frame AUPRC & F1 (fault=positive) ====")
if average_precision_score is None or f1_score is None:
    print("AUPRC/F1 계산을 위해 scikit-learn 설치가 필요합니다. (pip install scikit-learn)")
else:
    prob_np = prob.numpy()            # p_normal
    y_true_fault = 1 - true_np        # 1=고장
    y_pred_fault = 1 - pred_np
    y_score_fault = 1.0 - prob_np
    for m in range(M):
        y_true_flat = y_true_fault[:, :, m].ravel()
        y_pred_flat = y_pred_fault[:, :, m].ravel()
        y_score_flat= y_score_fault[:, :, m].ravel()
        auprc = average_precision_score(y_true_flat, y_score_flat) if y_true_flat.sum()>0 else 1.0
        f1_fault = f1_score(y_true_flat, y_pred_flat, pos_label=1) if (y_true_flat.sum()>0) else 1.0
        print(f"[m={m}] AUPRC={auprc:.4f}  F1={f1_fault:.4f}")

# (F) 세그먼트 IoU (any-motor & per-motor)
ious_any = []
per_motor_ious = [[] for _ in range(M)]
for i in range(N):
    t_true_any = _first_fault_onset_TxM(true_np[i])
    t_pred_any = _first_fault_onset_TxM(pred_np[i])
    ious_any.append(_segment_iou_from_onset(t_true_any, t_pred_any, T))
    for m in range(M):
        t_true_m = _first_fault_onset_1d(true_np[i, :, m])
        t_pred_m = _first_fault_onset_1d(pred_np[i, :, m])
        per_motor_ious[m].append(_segment_iou_from_onset(t_true_m, t_pred_m, T))

print("\n==== [Extra] Segment IoU ====")
print(f"Any-motor IoU mean/med            : {np.mean(ious_any):.4f} / {np.median(ious_any):.4f}")
for m in range(M):
    arr = np.array(per_motor_ious[m])
    print(f"[m={m}] IoU mean/med               : {arr.mean():.4f} / {np.median(arr):.4f}")

# (G) Calibration: Brier Score / ECE
print("\n==== [Extra] Calibration ====")
if brier_score_loss is None:
    print("Brier/ECE 계산을 위해 scikit-learn 설치가 필요합니다. (pip install scikit-learn)")
else:
    p_fault = 1.0 - prob.numpy()  # (N,T,M)
    y_fault = 1 - true_np         # (N,T,M), 1=고장
    briers = []
    for m in range(M):
        briers.append(brier_score_loss(y_fault[:, :, m].ravel(), p_fault[:, :, m].ravel()))
    print(f"Brier (mean over motors)          : {np.mean(briers):.6f}")
    eces = []
    for m in range(M):
        eces.append(_expected_calibration_error(y_fault[:, :, m].ravel().astype(np.float32),
                                               p_fault[:, :, m].ravel().astype(np.float32), n_bins=15))
    print(f"ECE   (mean over motors)          : {np.mean(eces):.6f}")

# (H) 히스테리시스/런-필터 민감도 (임계/연속길이 스윕)
print("\n==== [Extra] Hysteresis Sensitivity (grid) ====")
thr_grid = [0.4, 0.5, 0.6]   # '정상 확률' 기준으로는 (1 - thr)를 cut으로 사용
run_grid = [1, 2, 3]
results = []

p_normal = prob.numpy()  # prob: P(normal) from earlier

# 안전용: 온셋 함수가 없으면 정의
if '_first_fault_onset_TxM' not in globals():
    def _first_fault_onset_TxM(mat_TxM: np.ndarray):
        fault_any = (mat_TxM == 0).any(axis=1)  # 0=fault
        return int(np.argmax(fault_any)) if fault_any.any() else None

for thr in thr_grid:
    for runK in run_grid:
        # pr_bin: 1=normal, 0=fault (정상확률이 1 - thr 이상이면 정상으로 간주)
        pr_bin = (p_normal >= (1.0 - thr)).astype(np.int32)

        # 런-필터(연속 0 길이 runK 미만 스파이크 억제)
        if runK > 1:
            pr_filt = pr_bin.copy()
            for i in range(N):
                for m in range(M):
                    mask_fault = run_filter_zero_mask(pr_filt[i, :, m], min_run=runK)
                    pr_filt[i, :, m] = np.where(mask_fault, 0, 1)
            pr_bin = pr_filt

        # 온셋 탐지 성능/지연
        tp = fp = fn = 0
        delays_s = []
        for i in range(N):
            t_true = _first_fault_onset_TxM(true_np[i])  # GT: 0=fault
            any_fault_pred = (pr_bin[i] == 0).any(axis=1)
            t_pred = int(np.argmax(any_fault_pred)) if any_fault_pred.any() else None

            if t_true is None:
                if t_pred is not None:
                    fp += 1
            else:
                if t_pred is None:
                    fn += 1
                else:
                    if (t_pred >= t_true - PRE_TOL) and (t_pred <= t_true + POST_TOL):
                        tp += 1
                        delays_s.append(max(t_pred - t_true, 0) * dt)
                    else:
                        fn += 1

        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        TTD = float(np.mean(delays_s)) if len(delays_s) > 0 else None  # time-to-detect mean

        # False Alarm per hour (정상 시퀀스에서 첫 오탐 여부 카운트 / 정상 총 시간[h])
        fa_count = 0
        normal_total_hours = 0.0
        for i in range(N):
            if _first_fault_onset_TxM(true_np[i]) is None:  # 완전 정상 시퀀스
                any_fault_pred = (pr_bin[i] == 0).any(axis=1)
                t_pred = int(np.argmax(any_fault_pred)) if any_fault_pred.any() else None
                if t_pred is not None:
                    fa_count += 1
                normal_total_hours += (T * dt) / 3600.0

        fa_per_hour = (fa_count / normal_total_hours) if normal_total_hours > 0 else 0.0

        results.append(dict(
            thr=thr, runK=runK,
            onset_F1=F1, onset_P=P, onset_R=R,
            TTD_mean=TTD, FA_per_hour=fa_per_hour
        ))

# 출력부: 중첩 f-string 제거
for r in results:
    ttd = r.get('TTD_mean', None)
    ttd_str = "NA" if ttd is None else f"{ttd:.3f}s"
    print(
        f"thr={r['thr']:.2f}, run={r['runK']}  |  "
        f"F1={r['onset_F1']:.3f}, P/R={r['onset_P']:.3f}/{r['onset_R']:.3f}, "
        f"TTD={ttd_str}, FA/h={r['FA_per_hour']:.6f}"
    )

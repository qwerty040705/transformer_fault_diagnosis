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
MIN_RUN_PRED  = 1     # 예측 스파이크 억제(연속 0 프레임 최소길이). 1=끄기, 2~3 권장 가능

# ====== Top-K 패턴 매칭 설정 ======
TOP_K         = 2     # Majority Top-2
HAMMING_TOL   = 0     # 패턴 일치 허용 해밍 거리(0~2 권장)
COVERAGE_TAU  = 0.8   # 상위 K 패턴이 전체 프레임에서 차지해야 할 최소 커버리지

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
cfg.setdefault("rope_base", 10000.0)  # RoPE base (체크포인트에 저장되어 있으면 동일값 유지 권장)

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
        """
        Precomputes cos,sin tables for RoPE.
        base: theta base (e.g., 1e4).
        """
        def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
            super().__init__()
            # dim은 per-head 차원(head_dim)
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
        """
        q,k: (B, H, T, D)
        cos,sin: (T, D)
        """
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
        sin = sin.unsqueeze(0).unsqueeze(0)

        def rotate(x):
            x_even = x[..., 0::2]
            x_odd  = x[..., 1::2]
            x_rot_even = x_even * cos[..., 0::2] - x_odd * sin[..., 0::2]
            x_rot_odd  = x_odd  * cos[..., 1::2] + x_even * sin[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = x_rot_even
            out[..., 1::2] = x_rot_odd
            return out

        return rotate(q), rotate(k)

    class RotaryEncoderLayer(nn.Module):
        """ RoPE 적용한 Transformer Encoder Layer (batch_first=True) """
        def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="relu", norm_first=True,
                     max_seq_len=1024, base=10000.0):
            super().__init__()
            assert d_model % nhead == 0, "d_model must be divisible by nhead for RoPE"
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
            # x: (B,T,D)
            B, T, D = x.size()
            H = self.nhead
            Hd = self.head_dim

            W = self.self_attn.in_proj_weight    # (3D, D)
            b = self.self_attn.in_proj_bias      # (3D,)
            # q, k, v projection
            q_proj = torch.addmm(b[:D],     x.view(-1, D), W[:D, :].t()).view(B, -1, D)
            k_proj = torch.addmm(b[D:2*D],  x.view(-1, D), W[D:2*D, :].t()).view(B, -1, D)
            v_proj = torch.addmm(b[2*D:],   x.view(-1, D), W[2*D:, :].t()).view(B, -1, D)

            def shape(xp):
                return xp.view(B, -1, H, Hd).transpose(1, 2)  # (B,H,T,Hd)

            q = shape(q_proj)
            k = shape(k_proj)
            v = shape(v_proj)

            cos, sin = self.rotary.get_cos_sin(T)  # (T,Hd)
            q_rot, k_rot = apply_rotary(q, k, cos, sin)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q_rot, k_rot, v,
                dropout_p=self.self_attn.dropout.p if self.training else 0.0
            )  # (B,H,T,Hd)

            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
            out = self.self_attn.out_proj(attn_output)                             # (B,T,D)
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

    # ---- 공통 모델 정의 ----
    class FaultDiagnosisTransformer(nn.Module):
        def __init__(
            self,
            input_dim: int = 24,
            d_model: int = 64,
            nhead: int = 8,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            output_dim: int = 8,
            max_seq_len: int = 1000,
            posenc: str = "learned",  # "learned" | "sincos" | "rope"
            rope_base: float = 10000.0,
        ):
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
                self.use_additive_pos = False  # RoPE는 additive pos 안씀
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

            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, _ = x.shape
            if T > self.max_seq_len:
                raise ValueError(
                    f"T={T} exceeds max_seq_len={self.max_seq_len}. "
                    f"Increase max_seq_len when constructing the model."
                )
            z = self.input_proj(x) / math.sqrt(self.d_model)  # (B,T,d_model)
            if self.use_additive_pos:
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
# posenc 상이 시 strict=False 로드 (sincos/rope)
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
eq_frame = (true_np == pred_np).all(axis=2)   # (N,T) 모든 모터 일치해야 True
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
    T, M = gt_TxM.shape
    best_acc, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            pr_slice = pr_TxM[-lag:, :]
            gt_slice = gt_TxM[:T+lag, :]
        elif lag > 0:
            pr_slice = pr_TxM[:T-lag, :]
            gt_slice = gt_TxM[lag:, :]
        else:
            pr_slice = pr_TxM
            gt_slice = gt_TxM
        if pr_slice.size == 0:
            continue
        eq = (gt_slice == pr_slice).all(axis=1)  # (T') 프레임별 전체-모터 일치
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
                protos[idx] = (p, c + 1)
                assigned = True
                break
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
                used[j] = True
                found = True
                break
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
            if not idxs:
                return None, None
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

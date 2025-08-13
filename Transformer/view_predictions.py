# Transformer/view_predictions.py
import os
import math
import numpy as np
import torch
import torch.nn as nn

# 👉 GUI 없이도 그릴 수 있도록 백엔드 고정 (반드시 pyplot보다 먼저)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader, random_split

# ── Config ─────────────────────────────────────────────
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
seed       = 42

# ── 안전 로딩 헬퍼 ─────────────────────────────────────
def load_checkpoint_safely(ckpt_path, map_location):
    """
    PyTorch 2.6의 weights_only=True 안전 로딩과 호환되도록 allowlist를 추가하고,
    실패 시 신뢰 가능한 경우 weights_only=False로 폴백.
    """
    from torch.serialization import add_safe_globals
    try:
        # numpy 내부 타입들 허용(환경에 따라 필요)
        from numpy._core import multiarray as _ma
        add_safe_globals([
            np.dtype, np.ndarray, _ma._reconstruct,
            # dtypes.*DType 계열
            np.dtypes.Float32DType, np.dtypes.Float64DType,
            np.dtypes.Int64DType, np.dtypes.Int32DType, np.dtypes.BoolDType,
        ])
    except Exception:
        pass

    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception as e1:
        print("[WARN] Safe load (weights_only=True) failed:", repr(e1))
        print("       Falling back to weights_only=False (only if checkpoint is trusted).")
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)

# ── 입력 ───────────────────────────────────────────────
try:
    link_count = int(input("How many links?: ").strip())
except Exception:
    link_count = 1
    print("[WARN] Invalid input. Fallback to link_count=1")

try:
    sample_idx_to_plot = input("Which validation sample index? (default=0): ").strip()
    sample_idx_to_plot = int(sample_idx_to_plot) if sample_idx_to_plot != "" else 0
except Exception:
    sample_idx_to_plot = 0

# ── Paths ──────────────────────────────────────────────
data_path = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")
ckpt_path = os.path.join(repo_root, "Transformer", f"Transformer_link_{link_count}.pth")
out_dir   = os.path.join(repo_root, "Transformer", "vis")
os.makedirs(out_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────
data    = np.load(data_path)
desired = data["desired"]                      # (S,T,4,4)
actual  = data["actual"]
labels  = data["label"].astype(np.float32)     # (S,T,M)  1=정상, 0=고장
dt      = float(data.get("dt", 0.01))

S, T, _, _ = desired.shape
M = labels.shape[2]

# 입력 피처 (S,T,24)
des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)
X  = np.concatenate([des_12, act_12], axis=2).astype(np.float32)

# ── Load checkpoint safely ────────────────────────────
ckpt = load_checkpoint_safely(ckpt_path, map_location=device)

mean, std  = ckpt["train_mean"], ckpt["train_std"]
if isinstance(mean, torch.Tensor): mean = mean.cpu().numpy()
if isinstance(std, torch.Tensor):  std  = std.cpu().numpy()

cfg        = dict(ckpt.get("cfg", {}))
cfg.setdefault("d_model", 64)
cfg.setdefault("nhead", 8)
cfg.setdefault("num_layers", 2)
cfg.setdefault("dim_feedforward", 128)
cfg.setdefault("dropout", 0.1)
cfg.setdefault("posenc", "learned")   # "learned" | "sincos" | "rope"
cfg.setdefault("rope_base", 10000.0)

assert (ckpt["input_dim"], ckpt["T"], ckpt["M"]) == (24, T, M), \
    f"shape mismatch: ckpt has (24,{ckpt['T']},{ckpt['M']}) vs data (24,{T},{M})"

# ── Normalize & val split ─────────────────────────────
Xn = (X - mean) / std
torch.manual_seed(seed); np.random.seed(seed)
ds_all = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(labels))
train_sz = int(0.8 * S); val_sz = S - train_sz
_, val_ds = random_split(ds_all, [train_sz, val_sz],
                         generator=torch.Generator().manual_seed(seed))
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# ── Model (learned/sincos/rope 지원) ──────────────────
def build_model(input_dim, output_dim, max_seq_len, cfg):
    posenc   = cfg.get("posenc", "learned")
    rope_base= cfg.get("rope_base", 10000.0)

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
            super().__init__()
            inv_freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            t = torch.arange(max_seq_len, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            cos = torch.zeros(max_seq_len, dim)
            sin = torch.zeros(max_seq_len, dim)
            cos[:, 0::2] = torch.cos(freqs); cos[:, 1::2] = torch.cos(freqs)
            sin[:, 0::2] = torch.sin(freqs); sin[:, 1::2] = torch.sin(freqs)
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)
        def get_cos_sin(self, T: int): return self.cos_cached[:T], self.sin_cached[:T]

    def apply_rotary(q, k, cos, sin):
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        def rotate(x):
            xe, xo = x[..., 0::2], x[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = xe * cos[..., 0::2] - xo * sin[..., 0::2]
            out[..., 1::2] = xo * cos[..., 1::2] + xe * sin[..., 1::2]
            return out
        return rotate(q), rotate(k)

    class RotaryEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward, dropout, max_seq_len, base):
            super().__init__()
            assert d_model % nhead == 0
            self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.ff1 = nn.Linear(d_model, dim_feedforward)
            self.ff2 = nn.Linear(dim_feedforward, d_model)
            self.drop = nn.Dropout(dropout)
            self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
            self.act = nn.ReLU()
            self.nhead = nhead; self.hd = d_model // nhead
            self.rotary = RotaryEmbedding(self.hd, max_seq_len=max_seq_len, base=base)
        def forward(self, x):
            B,T,D = x.shape; H, Hd = self.nhead, self.hd
            W, b  = self.mha.in_proj_weight, self.mha.in_proj_bias
            q = torch.addmm(b[:D],     x.view(-1, D), W[:D, :].t()).view(B, T, D)
            k = torch.addmm(b[D:2*D],  x.view(-1, D), W[D:2*D, :].t()).view(B, T, D)
            v = torch.addmm(b[2*D:],   x.view(-1, D), W[2*D:, :].t()).view(B, T, D)
            def shape(t): return t.view(B, T, H, Hd).transpose(1, 2)  # (B,H,T,Hd)
            q, k, v = shape(q), shape(k), shape(v)
            cos, sin = self.rotary.get_cos_sin(T)
            q, k = apply_rotary(q, k, cos, sin)
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.mha.dropout.p if self.training else 0.0
            ).transpose(1,2).contiguous().view(B, T, D)
            x = x + self.mha.out_proj(attn)
            x = self.n1(x)
            x = self.n2(x + self.drop(self.ff2(self.act(self.ff1(x)))))
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

            if posenc in ("learned", "sincos"):
                if posenc == "learned":
                    self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
                    nn.init.normal_(self.pos_embedding, std=0.02)
                else:
                    pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
                    pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
                    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
                    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
                    self.register_buffer("pos_embedding", pe.unsqueeze(0), persistent=False)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, activation="relu", batch_first=True, norm_first=True
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
                self.use_additive_pos = True
            elif posenc == "rope":
                self.use_additive_pos = False
                self.encoder = nn.Sequential(*[
                    RotaryEncoderLayer(d_model, nhead, dim_feedforward, dropout, max_seq_len, rope_base)
                    for _ in range(num_layers)
                ])
            else:
                raise ValueError(f"Unknown posenc: {posenc}")

            self.pos_drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(d_model, output_dim)
            )
            nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

        def forward(self, x):
            B,T,_ = x.shape
            if T > self.max_seq_len:
                raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}.")
            z = self.input_proj(x) / math.sqrt(self.d_model)
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

# ── Build & load model ────────────────────────────────
model = build_model(24, M, T, cfg)
strict_load = True
if cfg.get("posenc", "learned") != "learned":
    strict_load = False
missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=strict_load)
if not strict_load:
    print(f"[INFO] Loaded with strict=False. missing={len(missing)}, unexpected={len(unexpected)}")
model.eval()

# ── Inference (validation) ────────────────────────────
sigmoid = nn.Sigmoid()
all_probs, all_preds, all_trues = [], [], []
with torch.no_grad():
    for xb, yb in DataLoader(val_ds, batch_size=batch_size, shuffle=False):
        logits = model(xb.to(device))
        prob   = sigmoid(logits).cpu()
        pred   = (prob >= 0.5).int()
        all_probs.append(prob); all_preds.append(pred); all_trues.append(yb.int())

prob = torch.cat(all_probs, 0).numpy()   # (N,T,M) p(normal)
pred = torch.cat(all_preds, 0).numpy()   # (N,T,M) 1/0
true = torch.cat(all_trues, 0).numpy()   # (N,T,M) 1/0
N = true.shape[0]

# ── Plot & Save ───────────────────────────────────────
def save_labels_and_preds(sample_idx: int):
    assert 0 <= sample_idx < N
    t = np.arange(T) * dt
    fig, axes = plt.subplots(M + 1, 1, figsize=(14, 2*(M+1)), sharex=True)
    for m in range(M):
        axes[m].step(t, true[sample_idx,:,m], where="post", linewidth=2, label="Label")
        axes[m].step(t, pred[sample_idx,:,m], where="post", linestyle="--", label="Pred")
        axes[m].set_ylim(-0.2, 1.2); axes[m].set_yticks([0,1]); axes[m].set_ylabel(f"m{m}")
        axes[m].grid(True, linestyle="--", alpha=0.4)
        if m == 0: axes[m].legend(loc="upper right")
    any_true = (true[sample_idx]==0).any(axis=1).astype(int)
    any_pred = (pred[sample_idx]==0).any(axis=1).astype(int)
    axes[-1].step(t, any_true, where="post", linewidth=2, label="any(Label)")
    axes[-1].step(t, any_pred, where="post", linestyle="--", label="any(Pred)")
    axes[-1].set_ylim(-0.2,1.2); axes[-1].set_yticks([0,1]); axes[-1].set_ylabel("any fault"); axes[-1].set_xlabel("Time (s)")
    axes[-1].grid(True, linestyle="--", alpha=0.4); axes[-1].legend(loc="upper right")
    plt.suptitle(f"Validation sample {sample_idx} — Labels vs Predictions")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"labels_preds_link{link_count}_sample{sample_idx}.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print("📁 Saved:", out_png)

def save_labels_only(sample_idx: int, motors=None):
    assert 0 <= sample_idx < N
    motors = range(M) if motors is None else motors
    t = np.arange(T) * dt
    fig, axes = plt.subplots(len(motors), 1, figsize=(14, 2*len(motors)), sharex=True)
    if not isinstance(axes, (list, np.ndarray)): axes = [axes]
    for ax, m in zip(axes, motors):
        ax.step(t, true[sample_idx,:,m], where="post", linewidth=2)
        ax.set_ylim(-0.2,1.2); ax.set_yticks([0,1]); ax.set_ylabel(f"m{m}")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(f"Validation sample {sample_idx} — Label time series")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"labels_only_link{link_count}_sample{sample_idx}.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print("📁 Saved:", out_png)

# ── 실행 ──────────────────────────────────────────────
# 필요에 따라 둘 중 하나 또는 둘 다 호출
save_labels_and_preds(sample_idx_to_plot)
# save_labels_only(sample_idx_to_plot)
print("✅ Done.")

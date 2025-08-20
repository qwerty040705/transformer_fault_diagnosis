# /home/rvl/transformer_fault_diagnosis/Transformer/view_predictions.py
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader, random_split

# ───────────────────── 기본 설정 ─────────────────────
repo_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
seed       = 42

# ======================================================================
# RoPE & Transformer (train_fault_transformer.py와 동일)
# ======================================================================
def _build_rope_cache(max_seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=None):
    if device is None: device = torch.device("cpu")
    if dtype is None:  dtype = torch.float32
    idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq = base ** (-idx / head_dim)
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("t,f->tf", t, inv_freq)
    sin = torch.zeros((max_seq_len, head_dim), device=device, dtype=dtype)
    cos = torch.zeros((max_seq_len, head_dim), device=device, dtype=dtype)
    sin[:, 0::2] = torch.sin(freqs);  sin[:, 1::2] = torch.sin(freqs)
    cos[:, 0::2] = torch.cos(freqs);  cos[:, 1::2] = torch.cos(freqs)
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    return cos, sin

def _rotate_half(x: torch.Tensor):
    x_even = x[..., 0::2]; x_odd  = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (x * cos) + (_rotate_half(x) * sin)

class EncoderBlockRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead; self.head_dim = d_model // nhead
        self.Wq = nn.Linear(d_model, d_model); self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model); self.Wo = nn.Linear(d_model, d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope_cache):
        B, T, D = x.shape; H, Dh = self.nhead, self.head_dim
        cos, sin = rope_cache
        h = self.ln1(x)
        q = self.Wq(h); k = self.Wk(h); v = self.Wv(h)

        def split_heads(t): return t.view(B, T, H, Dh)
        q = split_heads(q); k = split_heads(k); v = split_heads(v)
        q = apply_rope(q, cos[:, :T], sin[:, :T]); k = apply_rope(k, cos[:, :T], sin[:, :T])
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_attn.p if self.training else 0.0)
        x = x + self.Wo(attn.transpose(1,2).contiguous().view(B, T, D))
        x = x + self.dropout_ffn(self.ffn(self.ln2(x)))
        return x

class FaultDiagnosisTransformer(nn.Module):
    def __init__(self, input_dim=42, d_model=64, nhead=8, num_layers=2,
                 dim_feedforward=128, dropout=0.1, output_dim=8,
                 max_seq_len=1000, rope_base=10000.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model; self.head_dim = d_model // nhead
        self.max_seq_len = max_seq_len; self.rope_base = rope_base

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlockRoPE(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(d_model, output_dim))
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def _maybe_build_rope_cache(self, T, device, dtype):
        if self.rope_cos.numel()==0 or self.rope_cos.shape[1]<T or self.rope_cos.device!=device or self.rope_cos.dtype!=dtype:
            self.rope_cos, self.rope_sin = _build_rope_cache(T, self.head_dim, self.rope_base, device, dtype)

    def forward(self, x):
        B, T, _ = x.shape
        if T > self.max_seq_len: raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}")
        self._maybe_build_rope_cache(T, x.device, x.dtype)
        z = self.input_proj(x)/math.sqrt(self.d_model); z = self.pos_drop(z)
        for blk in self.blocks: z = blk(z, (self.rope_cos, self.rope_sin))
        return self.head(z)

# ======================================================================
# Feature helpers (train과 동일)
# ======================================================================
def mat_to_pR(T):  return T[..., :3, 3], T[..., :3, :3]

def _vee_skew(A):
    return np.stack([A[...,2,1]-A[...,1,2], A[...,0,2]-A[...,2,0], A[...,1,0]-A[...,0,1]], axis=-1)/2.0

def so3_log(Rm):
    tr = np.clip((np.einsum('...ii', Rm) - 1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(tr); A = Rm - np.swapaxes(Rm, -1, -2)
    v = _vee_skew(A); sin_th = np.sin(theta); eps = 1e-9
    scale = np.where(np.abs(sin_th)[...,None]>eps, (theta/(sin_th+eps))[...,None], np.ones_like(theta)[...,None])
    w = v*scale
    return np.where((theta<1e-6)[...,None], v, w)

def rotation_error_vec(R_des, R_act):
    return so3_log(np.matmul(np.swapaxes(R_des,-1,-2), R_act))

def rel_log_increment(R):
    S, T, _, _ = R.shape
    out = np.zeros((S,T,3), dtype=R.dtype)
    if T>1:
        R_rel = np.matmul(np.swapaxes(R[:,:-1],-1,-2), R[:,1:])
        out[:,1:,:] = so3_log(R_rel)
    return out

def time_diff(x):
    d = np.zeros_like(x); d[:,1:,...] = x[:,1:,...] - x[:,:-1,...]; return d

# ======================================================================
# 안전 체크포인트 로드
# ======================================================================
def load_checkpoint_safely(ckpt_path, map_location):
    from torch.serialization import add_safe_globals
    try:
        from numpy._core import multiarray as _ma
        add_safe_globals([np.dtype, np.ndarray, _ma._reconstruct,
                          np.dtypes.Float32DType, np.dtypes.Float64DType,
                          np.dtypes.Int64DType, np.dtypes.Int32DType, np.dtypes.BoolDType])
    except Exception:
        pass
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception as e1:
        print("[WARN] Safe load failed:", repr(e1))
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)

# ======================================================================
# 메인
# ======================================================================
if __name__ == "__main__":
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

    data_path = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")
    ckpt_path = os.path.join(repo_root, "Transformer", f"Transformer_link_{link_count}.pth")
    out_dir   = os.path.join(repo_root, "Transformer", "vis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] data_path: {data_path}")
    print(f"[INFO] ckpt_path: {ckpt_path}")
    print(f"[INFO] out_dir  : {out_dir}")

    data    = np.load(data_path)
    desired = data["desired"]
    actual  = data["actual"]
    labels  = data["label"].astype(np.float32)   # npz: 1=정상, 0=고장
    dt      = float(data.get("dt", 0.01))

    S, T, _, _ = desired.shape
    M = labels.shape[2]
    print(f"[INFO] Dataset: S={S}, T={T}, M={M}, dt={dt}")

    ckpt = load_checkpoint_safely(ckpt_path, map_location=device)
    mean = ckpt["train_mean"];  std = ckpt["train_std"]
    if isinstance(mean, torch.Tensor): mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):  std  = std.cpu().numpy()
    cfg = ckpt.get("cfg", {})
    input_dim = int(ckpt.get("input_dim", 42))

    # 42D 피처 (학습과 동일)
    des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
    act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)
    p_des, R_des = mat_to_pR(desired); p_act, R_act = mat_to_pR(actual)
    p_err  = p_act - p_des; r_err  = rotation_error_vec(R_des, R_act)
    dp_des = time_diff(p_des); dp_act = time_diff(p_act)
    dr_des = rel_log_increment(R_des); dr_act = rel_log_increment(R_act)
    X = np.concatenate([des_12, act_12, p_err, r_err, dp_des, dp_act, dr_des, dr_act], axis=2).astype(np.float32)
    assert X.shape[2] == input_dim, f"feature dim={X.shape[2]} (expected {input_dim})"

    Xn = (X - mean) / (std + 1e-9)

    torch.manual_seed(seed); np.random.seed(seed)
    full_ds  = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(labels))
    train_sz = int(0.8 * S); val_sz = S - train_sz
    _, val_ds = random_split(full_ds, [train_sz, val_sz], generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FaultDiagnosisTransformer(
        input_dim=input_dim,
        d_model=int(cfg.get("d_model", 64)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 2)),
        dim_feedforward=int(cfg.get("dim_feedforward", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        output_dim=M,
        max_seq_len=int(ckpt.get("T", T)),
        rope_base=float(cfg.get("rope_base", 10000.0)),
    ).to(device)

    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    sigmoid = nn.Sigmoid()
    probs, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            prob_y1 = sigmoid(logits).cpu().numpy()   # 1=고장 확률
            probs.append(prob_y1); trues.append(yb.numpy())

    p_fault = np.concatenate(probs, 0)   # (N,T,M)
    true_np = np.concatenate(trues, 0)   # (N,T,M)  npz: 1=정상
    true_fault = 1.0 - true_np           # (N,T,M)  1=고장
    N = true_fault.shape[0]
    print(f"[INFO] Validation N={N}")

    # ───────────── Helper: 공통 스타일 ─────────────
    def _style_axis(ax, m_label=None):
        ax.set_ylim(-0.05, 1.05)
        if m_label is not None: ax.set_ylabel(m_label)
        ax.grid(True, ls="--", alpha=0.5)

    # ───────────── 그림 1: 라벨 vs 확률 ─────────────
    def plot_probabilities(idx: int):
        assert 0 <= idx < N
        t = np.arange(T) * dt
        fig, axes = plt.subplots(M+1, 1, figsize=(14, 2*(M+1)), sharex=True)
        for m in range(M):
            axes[m].plot(t, p_fault[idx,:,m], lw=2, label="p(fault)")
            axes[m].step(t, true_fault[idx,:,m], where="post", lw=1.6, ls="--", label="Label(=fault)")
            _style_axis(axes[m], f"m{m}")
            if m == 0: axes[m].legend(loc="upper right")

        any_true = (true_fault[idx]==1).any(axis=1).astype(float)
        any_p    = p_fault[idx].max(axis=1)
        axes[-1].plot(t, any_p, lw=2, label="max p(fault)")
        axes[-1].step(t, any_true, where="post", lw=1.6, ls="--", label="any(Label=1)")
        _style_axis(axes[-1], "any"); axes[-1].set_xlabel("Time (s)")
        axes[-1].legend(loc="upper right")

        plt.suptitle(f"Validation sample {idx} — Probabilities vs Labels")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"probabilities_link{link_count}_sample{idx}.png")
        plt.savefig(out_png, dpi=200); plt.close(fig)
        print("📁 Saved:", out_png)

    # ───────────── 이진화 유틸 ─────────────
    def _binarize_with_hysteresis(p_seq, thr_on=0.6, thr_off=0.4):
        T_ = p_seq.shape[0]; out = np.zeros(T_, dtype=bool); cur = False
        for t in range(T_):
            p = p_seq[t]
            if not cur:
                if p >= thr_on: cur = True
            else:
                if p < thr_off: cur = False
            out[t] = cur
        return out

    def _prune_short_segments(sig, min_len=8):
        T_ = sig.shape[0]; out = sig.copy(); t=0
        while t < T_:
            if out[t]:
                s=t
                while t<T_ and out[t]: t+=1
                e=t
                if (e-s)<min_len: out[s:e]=False
            else:
                t+=1
        return out

    # ───────────── 그림 2: 라벨 vs 이진화 ─────────────
    def plot_binarized(idx: int, thr_on=0.6, thr_off=0.4, min_len=8):
        assert 0 <= idx < N
        t = np.arange(T) * dt
        bin_pred = np.zeros((T, M), dtype=bool)
        for m in range(M):
            b = _binarize_with_hysteresis(p_fault[idx,:,m], thr_on=thr_on, thr_off=thr_off)
            b = _prune_short_segments(b, min_len=min_len)
            bin_pred[:, m] = b

        fig, axes = plt.subplots(M+1, 1, figsize=(14, 2*(M+1)), sharex=True)
        for m in range(M):
            axes[m].step(t, bin_pred[:,m].astype(float), where="post", lw=2.0, label="Pred(bin)")
            axes[m].step(t, true_fault[idx,:,m], where="post", lw=1.6, ls="--", label="Label(=fault)")
            _style_axis(axes[m], f"m{m}")
            if m == 0: axes[m].legend(loc="upper right")

        any_true = (true_fault[idx]==1).any(axis=1).astype(float)
        any_pred = bin_pred.any(axis=1).astype(float)
        axes[-1].step(t, any_pred, where="post", lw=2.0, label="any Pred(bin)")
        axes[-1].step(t, any_true, where="post", lw=1.6, ls="--", label="any(Label=1)")
        _style_axis(axes[-1], "any"); axes[-1].set_xlabel("Time (s)")
        axes[-1].legend(loc="upper right")

        plt.suptitle(f"Validation sample {idx} — Binarized vs Labels (on={thr_on}, off={thr_off}, min_len={min_len})")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"binarized_link{link_count}_sample{idx}.png")
        plt.savefig(out_png, dpi=200); plt.close(fig)
        print("📁 Saved:", out_png)

    # ───────────── 실행 ─────────────
    plot_probabilities(sample_idx_to_plot)
    plot_binarized(sample_idx_to_plot, thr_on=0.6, thr_off=0.4, min_len=8)
    print("✅ Done.")

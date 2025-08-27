# /home/rvl/transformer_fault_diagnosis/Transformer/view_predictions_new.py
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
device     = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
batch_size = 16
seed       = 42

# ======================================================================
# RoPE & Transformer (train_new와 동일, in-place 제거)
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
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(inplace=False),
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
                 max_seq_len=2000, rope_base=10000.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model; self.head_dim = d_model // nhead
        self.max_seq_len = max_seq_len; self.rope_base = rope_base

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlockRoPE(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=False),
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
# Feature builders: {T_01..T_(N-1)N, T_0N} × 42D
# ======================================================================
def _mat_to_pR(T):  # (...,4,4) -> (...,3), (...,3,3)
    return T[..., :3, 3], T[..., :3, :3]

def _vee_skew(A):
    return np.stack([A[...,2,1]-A[...,1,2],
                     A[...,0,2]-A[...,2,0],
                     A[...,1,0]-A[...,0,1]], axis=-1) / 2.0

def _so3_log(Rm):
    tr = np.clip((np.einsum('...ii', Rm) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    A = Rm - np.swapaxes(Rm, -1, -2)
    v = _vee_skew(A)
    sin_th = np.sin(theta); eps = 1e-9
    scale = np.where(np.abs(sin_th)[...,None] > eps, (theta / (sin_th + eps))[...,None], np.ones_like(theta)[...,None])
    w = v * scale
    return np.where((theta < 1e-6)[...,None], v, w)

def _rot_err_vec(R_des, R_act):
    R_rel = np.matmul(np.swapaxes(R_des, -1, -2), R_act)
    return _so3_log(R_rel)

def _rel_log_increment_over_time(R):         # (S,T,K,3,3) -> (S,T,K,3)
    S,T,K = R.shape[:3]
    out = np.zeros((S,T,K,3), dtype=R.dtype)
    if T > 1:
        R_prev = R[:, :-1]
        R_next = R[:, 1:]
        R_rel  = np.matmul(np.swapaxes(R_prev, -1, -2), R_next)
        out[:, 1:, :, :] = _so3_log(R_rel)
    return out

def _time_diff_over_time(x):                 # (S,T,K,3) -> (S,T,K,3)
    d = np.zeros_like(x)
    if x.shape[1] > 1:
        d[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
    return d

def _flatten_3x4(T):                         # (...,4,4) -> (...,12)
    return T[..., :3, :4].reshape(*T.shape[:-2], 12)

def build_features_rel_plus_ee(d_rel, a_rel, d_cum, a_cum):
    """
    Inputs:
      d_rel, a_rel: (S,T,L,4,4)   relatives T_{i-1,i}
      d_cum, a_cum: (S,T,L,4,4)   cumulative; use last for EE T_{0->N}
    Return:
      X: (S,T, 42*(L+1)) where per-transform 42D =
         [des12|act12|p_err|r_err|dp_des|dp_act|dr_des|dr_act]
    """
    S, T, L = d_rel.shape[:3]
    d_T0N = d_cum[:, :, L-1]                 # (S,T,4,4)
    a_T0N = a_cum[:, :, L-1]
    d_all = np.concatenate([d_rel, d_T0N[:, :, None, :, :]], axis=2)  # (S,T,L+1,4,4)
    a_all = np.concatenate([a_rel, a_T0N[:, :, None, :, :]], axis=2)

    des_12 = _flatten_3x4(d_all)            # (S,T,K,12)
    act_12 = _flatten_3x4(a_all)

    p_des, R_des = _mat_to_pR(d_all)        # (S,T,K,3), (S,T,K,3,3)
    p_act, R_act = _mat_to_pR(a_all)
    p_err  = p_act - p_des                   # (S,T,K,3)
    r_err  = _rot_err_vec(R_des, R_act)      # (S,T,K,3)
    dp_des = _time_diff_over_time(p_des)     # (S,T,K,3)
    dp_act = _time_diff_over_time(p_act)
    dr_des = _rel_log_increment_over_time(R_des)  # (S,T,K,3)
    dr_act = _rel_log_increment_over_time(R_act)

    feats = np.concatenate([des_12, act_12, p_err, r_err, dp_des, dp_act, dr_des, dr_act], axis=-1)  # (S,T,K,42)
    X = feats.reshape(S, T, -1).astype(np.float32)  # (S,T,42*(L+1))
    return X

# ======================================================================
# 안전 체크포인트 로드
# ======================================================================
def load_checkpoint_safely(ckpt_path, map_location):
    from torch.serialization import add_safe_globals
    try:
        from numpy._core import multiarray as _ma
        import numpy as _np
        add_safe_globals([_np.dtype, _np.ndarray, _ma._reconstruct,
                          _np.dtypes.Float32DType, _np.dtypes.Float64DType,
                          _np.dtypes.Int64DType, _np.dtypes.Int32DType, _np.dtypes.BoolDType])
    except Exception:
        pass
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception as e1:
        print("[WARN] Safe load failed:", repr(e1))
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)

# ======================================================================
# 히스테리시스/최소길이 이진화
# ======================================================================
def _binarize_with_hysteresis(p_seq, thr_on, thr_off=None):
    if thr_off is None:
        thr_off = max(0.2, thr_on - 0.2)
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

    # 임계치(옵션) 입력
    raw_thr_on  = input("thr_on? (Enter=ckpt/best): ").strip()
    raw_thr_off = input("thr_off? (Enter=auto thr_on-0.2): ").strip()
    raw_min_len = input("min_len? (default=8): ").strip()

    data_path = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")
    ckpt_path = os.path.join(repo_root, "Transformer", f"Transformer_link_{link_count}_relEE42.pth")
    if not os.path.exists(ckpt_path):
        alt = os.path.join(repo_root, "Transformer", f"Transformer_link_{link_count}.pth")
        if os.path.exists(alt): ckpt_path = alt
    out_dir   = os.path.join(repo_root, "Transformer", "vis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] data_path: {data_path}")
    print(f"[INFO] ckpt_path: {ckpt_path}")
    print(f"[INFO] out_dir  : {out_dir}")

    # ---- load dataset (relatives + cumulative) ----
    data = np.load(data_path, allow_pickle=True)
    d_rel = data["desired_link_rel"]     # (S,T,L,4,4)
    a_rel = data["actual_link_rel"]
    d_cum = data["desired_link_cum"]
    a_cum = data["actual_link_cum"]
    labels= data["label"].astype(np.float32)   # npz: 1=정상, 0=고장
    dt    = float(data.get("dt", 0.01))

    S, T, L = d_rel.shape[:3]
    M = labels.shape[2]
    assert M == 8 * L, f"M={M} but expected 8*L={8*L}"
    print(f"[INFO] Dataset: S={S}, T={T}, L={L}, M={M}, dt={dt}")

    # ---- build features 42*(L+1) and normalize with ckpt stats ----
    X = build_features_rel_plus_ee(d_rel, a_rel, d_cum, a_cum)  # (S,T,42*(L+1))
    FEAT_DIM = X.shape[2]

    ckpt = load_checkpoint_safely(ckpt_path, map_location=device)
    mean = ckpt["train_mean"];  std = ckpt["train_std"]
    if isinstance(mean, torch.Tensor): mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):  std  = std.cpu().numpy()
    cfg = ckpt.get("cfg", {})
    input_dim = int(ckpt.get("input_dim", FEAT_DIM))
    assert input_dim == FEAT_DIM, f"feature dim={FEAT_DIM} (expected {input_dim})"

    # 스윕된 임계치 사용 (우선순위: per-motor → any)
    best_thr_any = float(ckpt.get("best_thr_any", ckpt.get("best_thr", 0.5)))
    best_thr_vec = ckpt.get("best_thr_per_motor", None)
    if isinstance(best_thr_vec, torch.Tensor): best_thr_vec = best_thr_vec.cpu().numpy()
    if best_thr_vec is None or (hasattr(best_thr_vec, "shape") and best_thr_vec.shape[0] != M):
        thr_per_motor = np.full(M, best_thr_any, dtype=np.float32)
    else:
        thr_per_motor = np.asarray(best_thr_vec, dtype=np.float32)

    # CLI 입력이 있으면 덮어쓰기(모든 모터에 동일 적용)
    if raw_thr_on != "":
        thr_per_motor[:] = float(raw_thr_on)
    if raw_thr_off != "":
        thr_off_global = float(raw_thr_off)
    else:
        thr_off_global = None  # 모터별 on에 따라 자동

    min_len = int(raw_min_len) if raw_min_len != "" else 8

    print(f"[INFO] thresholds:")
    print(f"  best_thr_any={best_thr_any:.3f}")
    if thr_per_motor.ndim==1:
        print(f"  per-motor on (first 8): {np.round(thr_per_motor[:8],3)}{' ...' if M>8 else ''}")
    print(f"  min_len={min_len}  (CLI on={raw_thr_on or 'None'}, off={raw_thr_off or 'auto'}, ckpt_per_motor={'yes' if best_thr_vec is not None else 'no'})")

    Xn = (X - mean) / (std + 1e-9)

    # ---- split as in training ----
    torch.manual_seed(seed); np.random.seed(seed)
    full_ds  = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(labels))
    train_sz = int(0.8 * S); val_sz = S - train_sz
    _, val_ds = random_split(full_ds, [train_sz, val_sz], generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---- restore model ----
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
            on_m = float(thr_per_motor[m])
            off_m = max(0.2, on_m - 0.2) if thr_off_global is None else thr_off_global
            axes[m].plot(t, p_fault[idx,:,m], lw=2, label="p(fault)")
            axes[m].axhline(on_m,  ls="--", lw=1.0, color="k", alpha=0.5)
            axes[m].axhline(off_m, ls=":",  lw=1.0, color="k", alpha=0.5)
            axes[m].step(t, true_fault[idx,:,m], where="post", lw=1.6, ls="--", label="Label(=fault)")
            _style_axis(axes[m], f"m{m}")
            if m == 0: axes[m].legend(loc="upper right")

        any_true = (true_fault[idx]==1).any(axis=1).astype(float)
        any_p    = p_fault[idx].max(axis=1)
        axes[-1].plot(t, any_p, lw=2, label="max p(fault)")
        # any plot에는 대표 on/off로 best_thr_any 사용
        axes[-1].axhline(best_thr_any,          ls="--", lw=1.0, color="k", alpha=0.6)
        axes[-1].axhline(max(0.2, best_thr_any-0.2), ls=":",  lw=1.0, color="k", alpha=0.6)
        axes[-1].step(t, any_true, where="post", lw=1.6, ls="--", label="any(Label=1)")
        _style_axis(axes[-1], "any"); axes[-1].set_xlabel("Time (s)")
        axes[-1].legend(loc="upper right")

        plt.suptitle(f"Validation sample {idx} — Probabilities vs Labels (42x(L+1) features)")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"probabilities_relEE42_link{link_count}_sample{idx}.png")
        plt.savefig(out_png, dpi=200); plt.close(fig)
        print("📁 Saved:", out_png)

    # ───────────── 그림 2: 라벨 vs 이진화 ─────────────
    def plot_binarized(idx: int):
        assert 0 <= idx < N
        t = np.arange(T) * dt
        bin_pred = np.zeros((T, M), dtype=bool)
        for m in range(M):
            on_m = float(thr_per_motor[m])
            off_m = max(0.2, on_m - 0.2) if thr_off_global is None else thr_off_global
            b = _binarize_with_hysteresis(p_fault[idx,:,m], thr_on=on_m, thr_off=off_m)
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

        plt.suptitle(f"Validation sample {idx} — Binarized vs Labels "
                     f"(min_len={min_len}, per-motor on[0]={thr_per_motor[0]:.2f})")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"binarized_relEE42_link{link_count}_sample{idx}.png")
        plt.savefig(out_png, dpi=200); plt.close(fig)
        print("📁 Saved:", out_png)

        # 추가로 npy 저장 (원하면 분석에 사용)
        np.save(os.path.join(out_dir, f"p_fault_link{link_count}_sample{idx}.npy"), p_fault[idx])
        np.save(os.path.join(out_dir, f"pred_bin_link{link_count}_sample{idx}.npy"), bin_pred.astype(np.int8))
        np.save(os.path.join(out_dir, f"true_fault_link{link_count}_sample{idx}.npy"), true_fault[idx].astype(np.int8))

    # ───────────── 실행 ─────────────
    plot_probabilities(sample_idx_to_plot)
    plot_binarized(sample_idx_to_plot)
    print("✅ Done.")

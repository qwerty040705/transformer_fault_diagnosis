# /Users/dnbn/code/transformer_fault_diagnosis/Transformer/eval_fault_transformer_new.py
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

# ── Config ───────────────────────────────
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device     = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
batch_size = 64
seed       = 42
print("device:", device)

# ── RoPE utils ─────────────
def _build_rope_cache(max_seq_len, head_dim, base=10000.0, device=None, dtype=None):
    if device is None: device = torch.device("cpu")
    if dtype is None:  dtype = torch.float32
    idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq = base ** (-idx / head_dim)
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("t,f->tf", t, inv_freq)
    sin = torch.zeros((max_seq_len, head_dim), device=device, dtype=dtype)
    cos = torch.zeros((max_seq_len, head_dim), device=device, dtype=dtype)
    sin[:, 0::2] = torch.sin(freqs); sin[:, 1::2] = torch.sin(freqs)
    cos[:, 0::2] = torch.cos(freqs); cos[:, 1::2] = torch.cos(freqs)
    return cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)

def _rotate_half(x):
    x_even = x[..., 0::2]; x_odd  = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)

def apply_rope(x, cos, sin): return (x * cos) + (_rotate_half(x) * sin)

class EncoderBlockRoPE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead; self.head_dim = d_model // nhead
        self.Wq = nn.Linear(d_model, d_model); self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model); self.Wo = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(inplace=False),
                                 nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.dropout_attn = nn.Dropout(dropout); self.dropout_ffn = nn.Dropout(dropout)
    def forward(self, x, rope_cache):
        B,T,D = x.shape; H = self.nhead; Dh = self.head_dim
        cos, sin = rope_cache
        h = self.ln1(x)
        q = self.Wq(h).view(B,T,H,Dh); k = self.Wk(h).view(B,T,H,Dh); v = self.Wv(h).view(B,T,H,Dh)
        q = apply_rope(q, cos[:, :T], sin[:, :T]); k = apply_rope(k, cos[:, :T], sin[:, :T])
        q = q.transpose(1,2); k = k.transpose(1,2); v = v.transpose(1,2)
        attn = F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout_attn.p if self.training else 0.0)
        x = x + self.Wo(attn.transpose(1,2).contiguous().view(B,T,D))
        x = x + self.dropout_ffn(self.ffn(self.ln2(x)))
        return x

class FaultDiagnosisTransformer(nn.Module):
    def __init__(self, input_dim=42, d_model=64, nhead=8, num_layers=2,
                 dim_feedforward=128, dropout=0.1, output_dim=8,
                 max_seq_len=2000, rope_base=10000.0):
        super().__init__()
        assert d_model % nhead == 0
        self.max_seq_len = max_seq_len; self.d_model = d_model
        self.nhead = nhead; self.head_dim = d_model // nhead; self.rope_base = rope_base
        self.input_proj = nn.Linear(input_dim, d_model); self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlockRoPE(d_model,nhead,dim_feedforward,dropout)
                                     for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=False),
                                  nn.Dropout(dropout), nn.Linear(d_model, output_dim))
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)
    def _maybe_build_rope_cache(self, T, device, dtype):
        need = (self.rope_cos.numel()==0 or self.rope_cos.shape[1]<T
                or self.rope_cos.device!=device or self.rope_cos.dtype!=dtype)
        if need:
            self.rope_cos, self.rope_sin = _build_rope_cache(T, self.head_dim, self.rope_base, device, dtype)
    def forward(self, x):
        B,T,_=x.shape
        if T > self.max_seq_len: raise ValueError(f"T={T} > max_seq_len={self.max_seq_len}")
        self._maybe_build_rope_cache(T, x.device, x.dtype)
        z = self.input_proj(x)/math.sqrt(self.d_model); z = self.pos_drop(z)
        for blk in self.blocks: z = blk(z, (self.rope_cos, self.rope_sin))
        return self.head(z)

# ── Feature builder: {T_01..T_(N-1)N, T_0N} × 42D ─────
def _mat_to_pR(T):  return T[..., :3, 3], T[..., :3, :3]
def _vee_skew(A):   return np.stack([A[...,2,1]-A[...,1,2], A[...,0,2]-A[...,2,0], A[...,1,0]-A[...,0,1]], axis=-1)/2.0
def _so3_log(Rm):
    tr = np.clip((np.einsum('...ii', Rm) - 1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(tr); A = Rm - np.swapaxes(Rm, -1, -2)
    v = _vee_skew(A); sin_th = np.sin(theta); eps = 1e-9
    scale = np.where(np.abs(sin_th)[...,None]>eps, (theta/(sin_th+eps))[...,None], np.ones_like(theta)[...,None])
    w = v*scale
    return np.where((theta<1e-6)[...,None], v, w)
def _rot_err_vec(R_des, R_act): return _so3_log(np.matmul(np.swapaxes(R_des,-1,-2), R_act))
def _rel_log_increment_over_time(R):
    S,T,K = R.shape[:3]
    out = np.zeros((S,T,K,3), dtype=R.dtype)
    if T>1:
        R_rel = np.matmul(np.swapaxes(R[:,:-1],-1,-2), R[:,1:])
        out[:,1:,:] = _so3_log(R_rel)
    return out
def _time_diff_over_time(x):
    d=np.zeros_like(x)
    if x.shape[1]>1: d[:,1:]=x[:,1:]-x[:,:-1]
    return d
def _flatten_3x4(T): return T[..., :3, :4].reshape(*T.shape[:-2], 12)

def build_features_rel_plus_ee(d_rel, a_rel, d_cum, a_cum):
    S, T, L = d_rel.shape[:3]
    d_T0N = d_cum[:, :, L-1]; a_T0N = a_cum[:, :, L-1]
    d_all = np.concatenate([d_rel, d_T0N[:, :, None]], axis=2)
    a_all = np.concatenate([a_rel, a_T0N[:, :, None]], axis=2)
    des_12 = _flatten_3x4(d_all); act_12 = _flatten_3x4(a_all)
    p_des, R_des = _mat_to_pR(d_all); p_act, R_act = _mat_to_pR(a_all)
    p_err  = p_act - p_des; r_err = _rot_err_vec(R_des, R_act)
    dp_des = _time_diff_over_time(p_des); dp_act = _time_diff_over_time(p_act)
    dr_des = _rel_log_increment_over_time(R_des); dr_act = _rel_log_increment_over_time(R_act)
    feats = np.concatenate([des_12, act_12, p_err, r_err, dp_des, dp_act, dr_des, dr_act], axis=-1)  # (S,T,K,42)
    return feats.reshape(S, T, -1).astype(np.float32)

# ── Hysteresis + min-length binarizer ─────
def binarize_seq(p_seq, on, off=None, min_len=8):
    if off is None: off = max(0.2, on - 0.2)
    T_ = p_seq.shape[0]; out = np.zeros(T_, dtype=bool); cur=False
    for t in range(T_):
        p = p_seq[t]
        if not cur:
            cur = (p >= on)
        else:
            cur = (p >= off)
        out[t] = cur
    # prune short segments
    t=0
    while t<T_:
        if out[t]:
            s=t
            while t<T_ and out[t]: t+=1
            if (t-s)<min_len: out[s:t]=False
        else:
            t+=1
    return out

# ── Safe checkpoint load ─────────────
def load_checkpoint_safely(path, map_location):
    from torch.serialization import add_safe_globals
    try:
        from numpy._core import multiarray as _ma
        import numpy as _np
        add_safe_globals([_np.dtype, _np.ndarray, _ma._reconstruct,
                          _np.dtypes.Float32DType, _np.dtypes.Float64DType,
                          _np.dtypes.Int64DType,  _np.dtypes.Int32DType, _np.dtypes.BoolDType])
    except Exception:
        pass
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)

# ── Paths ────────────────────────────────
link_count = int(input("How many links?: ").strip())
data_path  = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")
ckpt_name  = f"Transformer_link_{link_count}_relEE42.pth"
ckpt_path  = os.path.join(repo_root, "Transformer", ckpt_name)
if not os.path.exists(ckpt_path):
    alt = os.path.join(repo_root, "Transformer", f"Transformer_link_{link_count}.pth")
    if os.path.exists(alt): ckpt_path = alt

# ── Load data ────────────────────────────
data = np.load(data_path, allow_pickle=True)
d_rel = data["desired_link_rel"]; a_rel = data["actual_link_rel"]
d_cum = data["desired_link_cum"]; a_cum = data["actual_link_cum"]
labels = data["label"].astype(np.float32)   # 1=정상, 0=고장
dt = float(data.get("dt", 0.01))
S, T, L = d_rel.shape[:3]
M = labels.shape[2]
assert M == 8 * L, f"M={M} but expected 8*L={8*L}"

# ── Build features ───────────────────────
X = build_features_rel_plus_ee(d_rel, a_rel, d_cum, a_cum)
FEAT_DIM = X.shape[2]
true_fault = (1.0 - labels).astype(np.float32)  # 1=고장

# ── Load checkpoint (safe) ───────────────
ckpt = load_checkpoint_safely(ckpt_path, map_location=device)
mean = ckpt["train_mean"]; std = ckpt["train_std"]
if isinstance(mean, torch.Tensor): mean = mean.cpu().numpy()
if isinstance(std, torch.Tensor):  std  = std.cpu().numpy()
cfg = ckpt.get("cfg", {})
ckpt_input_dim = int(ckpt.get("input_dim", FEAT_DIM))
if ckpt_input_dim != FEAT_DIM:
    raise ValueError(f"Input dim mismatch: ckpt={ckpt_input_dim}, built={FEAT_DIM}. "
                     f"(L in data={L}; expected FEAT_DIM=42*(L+1)={42*(L+1)})")

# thresholds (지원: any + per-motor)
best_thr_any  = float(ckpt.get("best_thr_any", ckpt.get("best_thr", 0.5)))
best_thr_vec  = ckpt.get("best_thr_per_motor", None)
if isinstance(best_thr_vec, torch.Tensor): best_thr_vec = best_thr_vec.cpu().numpy()
if best_thr_vec is None or (hasattr(best_thr_vec, "shape") and best_thr_vec.shape[0] != M):
    thr_per_motor = np.full(M, best_thr_any, dtype=np.float32)
else:
    thr_per_motor = np.asarray(best_thr_vec, dtype=np.float32)

# ── Normalize ────────────────────────────
X = (X - mean) / (std + 1e-9)

# ── Dataset / split ──────────────────────
torch.manual_seed(seed); np.random.seed(seed)
ds_all = TensorDataset(torch.from_numpy(X), torch.from_numpy(true_fault))
train_sz = int(0.8 * S); val_sz = S - train_sz
_, val_ds = random_split(ds_all, [train_sz, val_sz], generator=torch.Generator().manual_seed(seed))
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# ── Model ────────────────────────────────
model = FaultDiagnosisTransformer(
    input_dim=ckpt_input_dim,
    d_model=int(cfg.get("d_model", 64)),
    nhead=int(cfg.get("nhead", 8)),
    num_layers=int(cfg.get("num_layers", 2)),
    dim_feedforward=int(cfg.get("dim_feedforward", 128)),
    dropout=float(cfg.get("dropout", 0.1)),
    output_dim=M,
    max_seq_len=int(ckpt.get("T", T)),
    rope_base=float(cfg.get("rope_base", 10000.0)),
).to(device)
state = ckpt.get("model_state", ckpt)  # 호환
model.load_state_dict(state, strict=True)
model.eval()

# ── Inference ───────────────────────────
sigmoid = torch.nn.Sigmoid()
probs, trues = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        logits = model(xb.to(device))
        p = sigmoid(logits).cpu().numpy()  # 1=고장 확률
        probs.append(p); trues.append(yb.numpy())
p_fault = np.concatenate(probs, 0)   # (N,T,M)
true    = np.concatenate(trues, 0)   # (N,T,M), 1=고장
N, T, M = true.shape

# ── Post-process: hysteresis + min length (모터별 임계값) ──
min_len = 8
pred_bin = np.zeros_like(true, dtype=np.int32)
for i in range(N):
    for m in range(M):
        on  = float(thr_per_motor[m])
        off = max(0.2, on - 0.2)
        b = binarize_seq(p_fault[i,:,m], on=on, off=off, min_len=min_len)
        pred_bin[i,:,m] = b.astype(np.int32)

# ── Metrics ─────────────────────────────
eq_frame = (true == pred_bin).all(axis=2)
print("\n==== [Exact-all motors] Frame-wise accuracy ====")
print(f"Overall frame accuracy : {eq_frame.mean():.4f}")

# Any-fault frame accuracy
any_fault_frame_acc = np.zeros((N,T), dtype=float)
for i in range(N):
    for t in range(T):
        gt = (true[i,t]==1); pr=(pred_bin[i,t]==1)
        any_fault_frame_acc[i,t] = 1.0 if (gt.any()==pr.any() and (not gt.any() or (gt & pr).any())) else 0.0
print("\n==== [Any-fault frame accuracy] ====")
print(f"Any-fault overall : {any_fault_frame_acc.mean():.4f}")

# Onset detection (any motor) — within 0.05s tolerance
def first_fault_onset(mat_TxM):
    fault_any=(mat_TxM==1).any(axis=1)
    return int(np.argmax(fault_any)) if fault_any.any() else None

tp=fp=fn=0; delays=[]
tol_frames = int(round(0.05 / dt)) if dt>0 else 0
for i in range(N):
    t_true=first_fault_onset(true[i]); t_pred=first_fault_onset(pred_bin[i])
    if t_true is None:
        if t_pred is not None: fp+=1
    else:
        if t_pred is None:
            fn+=1
        else:
            if abs(t_pred-t_true) <= tol_frames:
                tp+=1; delays.append(max(t_pred-t_true,0)*dt)
            else:
                fn+=1
prec=tp/(tp+fp) if(tp+fp) else 0.0
rec =tp/(tp+fn) if(tp+fn) else 0.0
f1  =2*prec*rec/(prec+rec) if(prec+rec) else 0.0
print("\n==== [Onset detection (any-motor)] ====")
print(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
if delays:
    print(f"Onset delay mean={np.mean(delays):.4f}s median={np.median(delays):.4f}s")

# Per-motor PRF
tp_m = ( (true==1) & (pred_bin==1) ).sum(axis=(0,1))
fp_m = ( (true==0) & (pred_bin==1) ).sum(axis=(0,1))
fn_m = ( (true==1) & (pred_bin==0) ).sum(axis=(0,1))
prec_m = np.divide(tp_m, tp_m+fp_m, out=np.zeros_like(tp_m, dtype=float), where=(tp_m+fp_m)>0)
rec_m  = np.divide(tp_m, tp_m+fn_m, out=np.zeros_like(tp_m, dtype=float), where=(tp_m+fn_m)>0)
f1_m   = np.divide(2*prec_m*rec_m, prec_m+rec_m, out=np.zeros_like(prec_m), where=(prec_m+rec_m)>0)
print("\n==== [Per-motor PRF] ====")
print(f"macro-Precision={prec_m.mean():.4f} macro-Recall={rec_m.mean():.4f} macro-F1={f1_m.mean():.4f}")

print("\nDone.")

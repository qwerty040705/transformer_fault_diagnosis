# /home/rvl/transformer_fault_diagnosis/Transformer/eval_fault_transformer.py
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

# ── Config ───────────────────────────────
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
seed       = 42
print("device:", device)

# ── Feature builder (24 / 42) ────────────
def build_features(desired, actual, mode_dim=24):
    S, T, _, _ = desired.shape
    des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
    act_12 = actual[:, :, :3, :4].reshape(S, T, 12)
    if mode_dim == 24:
        return np.concatenate([des_12, act_12], axis=2).astype(np.float32)

    # === 42D 확장 ===
    def mat_to_pR(Tm): return Tm[..., :3, 3], Tm[..., :3, :3]
    def _vee_skew(A):
        return np.stack([A[...,2,1]-A[...,1,2],
                         A[...,0,2]-A[...,2,0],
                         A[...,1,0]-A[...,0,1]], axis=-1) / 2.0
    def so3_log(Rm):
        tr = np.clip((np.einsum('...ii', Rm)-1.0)/2.0, -1.0, 1.0)
        theta = np.arccos(tr); A = Rm - np.swapaxes(Rm, -1, -2)
        v = _vee_skew(A); sin_th = np.sin(theta); eps=1e-9
        scale = np.where(np.abs(sin_th)[...,None]>eps,
                         (theta/(sin_th+eps))[...,None], 1.0)
        w = v*scale
        return np.where((theta<1e-6)[...,None], v, w)
    def rotation_error_vec(R_des, R_act):
        return so3_log(np.matmul(np.swapaxes(R_des,-1,-2), R_act))
    def rel_log_increment(R):
        S_,T_,_,_=R.shape; out=np.zeros((S_,T_,3),dtype=R.dtype)
        if T_<=1: return out
        R_rel=np.matmul(np.swapaxes(R[:,:-1],-1,-2),R[:,1:])
        out[:,1:,:]=so3_log(R_rel); return out
    def time_diff(x):
        d=np.zeros_like(x); d[:,1:]=x[:,1:]-x[:,:-1]; return d

    p_des,R_des=mat_to_pR(desired); p_act,R_act=mat_to_pR(actual)
    p_err=p_act-p_des; r_err=rotation_error_vec(R_des,R_act)
    dp_des,dp_act=time_diff(p_des),time_diff(p_act)
    dr_des,dr_act=rel_log_increment(R_des),rel_log_increment(R_act)
    X42=np.concatenate([des_12,act_12,p_err,r_err,dp_des,dp_act,dr_des,dr_act],axis=2).astype(np.float32)
    return X42

# ── RoPE utils (훈련 스크립트와 동일) ─────────
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
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(inplace=True),
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
                 max_seq_len=1000, rope_base=10000.0):
        super().__init__()
        assert d_model % nhead == 0
        self.max_seq_len = max_seq_len; self.d_model = d_model
        self.nhead = nhead; self.head_dim = d_model // nhead; self.rope_base = rope_base
        self.input_proj = nn.Linear(input_dim, d_model); self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlockRoPE(d_model,nhead,dim_feedforward,dropout)
                                     for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
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

# ── Paths ────────────────────────────────
link_count = int(input("How many links?: ").strip())
data_path  = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")
ckpt_path  = os.path.join(repo_root, "Transformer", f"Transformer_link_{link_count}.pth")

# ── Load data ────────────────────────────
data    = np.load(data_path)
desired = data["desired"]; actual = data["actual"]
labels  = data["label"].astype(np.float32)  # 원본: 1=정상, 0=고장
dt      = float(data.get("dt",0.01))
S,T,_,_ = desired.shape
M       = labels.shape[2]

# ── Load checkpoint ─────────────────────
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
ckpt_input_dim = int(ckpt.get("input_dim", 42))
mean,std  = ckpt["train_mean"], ckpt["train_std"]
if isinstance(mean, torch.Tensor): mean=mean.cpu().numpy()
if isinstance(std, torch.Tensor):  std=std.cpu().numpy()
cfg = ckpt.get("cfg", {})

# ── Features & labels (라벨을 '고장=1'로 뒤집기) ─────
X = build_features(desired, actual, mode_dim=ckpt_input_dim)
true_fault = (1.0 - labels).astype(np.float32)    # 1=고장, 0=정상 (학습 타깃과 동일)
X = (X - mean) / (std + 1e-9)

# ── Dataset ─────────────────────────────
torch.manual_seed(seed); np.random.seed(seed)
ds_all = TensorDataset(torch.from_numpy(X), torch.from_numpy(true_fault))
train_sz=int(0.8*S); val_sz=S-train_sz
_, val_ds = random_split(ds_all,[train_sz,val_sz],generator=torch.Generator().manual_seed(seed))
val_loader = DataLoader(val_ds,batch_size=batch_size,shuffle=False)

# ── Model (훈련과 동일) ─────────────────
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
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()

# ── Inference ───────────────────────────
sigmoid = torch.nn.Sigmoid()
all_prob, all_pred, all_true = [], [], []
thr = 0.9  
with torch.no_grad():
    for xb, yb in val_loader:
        logits = model(xb.to(device))          # (B,T,M)
        p_fault = sigmoid(logits).cpu()        # 1=고장 확률
        pred    = (p_fault >= thr).int()
        all_prob.append(p_fault); all_pred.append(pred); all_true.append(yb.int())
prob = torch.cat(all_prob,0).numpy()   # (N,T,M)
pred = torch.cat(all_pred,0).numpy()   # (N,T,M)
true = torch.cat(all_true,0).numpy()   # (N,T,M), 1=고장

N, T, M = true.shape

# ── Metrics ─────────────────────────────
# (1) Frame-wise accuracy (all motors exact)
eq_frame = (true == pred).all(axis=2)
print("\n==== [Exact-all motors] Frame-wise accuracy ====")
print(f"Overall frame accuracy : {eq_frame.mean():.4f}")

# (2) Onset detection (any motor)
def first_fault_onset(mat_TxM):
    fault_any=(mat_TxM==1).any(axis=1)
    return int(np.argmax(fault_any)) if fault_any.any() else None

tp=fp=fn=0; delays=[]
for i in range(N):
    t_true=first_fault_onset(true[i]); t_pred=first_fault_onset(pred[i])
    if t_true is None:
        if t_pred is not None: fp+=1
    else:
        if t_pred is None: fn+=1
        else:
            # 허용 지연: 0.05s
            if abs(t_pred-t_true) <= int(0.05/dt):
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

# (3) Any-fault frame accuracy
any_fault_frame_acc=np.zeros((N,T))
for i in range(N):
    for t in range(T):
        gt = (true[i,t]==1); pr=(pred[i,t]==1)
        if not gt.any():
            any_fault_frame_acc[i,t] = 1.0 if not pr.any() else 0.0
        else:
            any_fault_frame_acc[i,t] = 1.0 if len(np.intersect1d(np.where(gt)[0], np.where(pr)[0]))>0 else 0.0
print("\n==== [Any-fault frame accuracy] ====")
print(f"Any-fault overall : {any_fault_frame_acc.mean():.4f}")


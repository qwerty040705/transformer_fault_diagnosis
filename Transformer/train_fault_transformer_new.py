import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

# ========================= RoPE utils =========================
def _build_rope_cache(max_seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=None):
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

def _rotate_half(x: torch.Tensor):
    x_even = x[..., 0::2]; x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (x * cos) + (_rotate_half(x) * sin)

# ==================== Transformer (RoPE) ======================
class EncoderBlockRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.Wq = nn.Linear(d_model, d_model); self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model); self.Wo = nn.Linear(d_model, d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.dropout_ffn = nn.Dropout(dropout); self.ln2 = nn.LayerNorm(d_model)
        for m in [self.Wq, self.Wk, self.Wv, self.Wo, self.ffn[0], self.ffn[3]]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, rope_cache):
        B, T, D = x.shape; H, Dh = self.nhead, self.head_dim
        cos, sin = rope_cache
        h = self.ln1(x)
        q = self.Wq(h).view(B, T, H, Dh)
        k = self.Wk(h).view(B, T, H, Dh)
        v = self.Wv(h).view(B, T, H, Dh)
        q = apply_rope(q, cos[:, :T], sin[:, :T])
        k = apply_rope(k, cos[:, :T], sin[:, :T])
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_attn.p if self.training else 0.0)
        x = x + self.Wo(attn.transpose(1, 2).contiguous().view(B, T, D))
        x = x + self.dropout_ffn(self.ffn(self.ln2(x)))
        return x

class FaultDiagnosisTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=8, num_layers=2,
                 dim_feedforward=128, dropout=0.1, max_seq_len=2000, rope_base=10000.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlockRoPE(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_model, output_dim))
        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)
        self.max_seq_len = max_seq_len; self.rope_base = rope_base
        self.head_dim = d_model // nhead; self.nhead = nhead

    def _maybe_build_rope_cache(self, T, device, dtype):
        need = (self.rope_cos.numel()==0) or (self.rope_cos.shape[1] < T) or (self.rope_cos.device!=device) or (self.rope_cos.dtype!=dtype)
        if need:
            self.rope_cos, self.rope_sin = _build_rope_cache(T, self.head_dim, self.rope_base, device, dtype)

    def forward(self, x):
        B, T, _ = x.shape
        if T > self.max_seq_len: raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}")
        self._maybe_build_rope_cache(T, x.device, x.dtype)
        z = self.input_proj(x) / math.sqrt(self.d_model)
        z = self.pos_drop(z)
        cache = (self.rope_cos, self.rope_sin)
        for blk in self.blocks: z = blk(z, cache)
        return self.head(z)

# ========== Exact-All loss (AND-softmin + group focal + FP hinge) ==========
def softmin_beta(x, beta, dim=-1):
    # softmin(x) = -1/beta * logsumexp(-beta*x)
    return -(1.0/beta) * torch.logsumexp(-beta * x, dim=dim)

def loss_exactall_and_only(logits, yb, beta=10.0, m0=1.5, gamma=2.0,
                           lambda_and=1.0, lambda_fp=0.05, fp_margin=0.1):
    """
    logits, yb: (B,T,M), yb∈{0,1}
    BCE는 완전 제외. Exact-All 기준만 학습.
    """
    y_pm = 2*yb - 1
    s = y_pm * logits                              # (B,T,M)

    p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)  # for weights/hinge
    q = torch.where(yb.bool(), p, 1.0 - p)         # (B,T,M)
    Q = q.prod(dim=2).detach()                     # (B,T) stop-grad
    w = (1.0 - Q).pow(gamma)                       # (B,T)

    softmin_s = softmin_beta(s, beta=beta, dim=2)  # (B,T)
    and_core = torch.relu(m0 - softmin_s)          # (B,T)
    L_and = (w * and_core).mean()

    is_normal = (yb.sum(dim=2) == 0)               # (B,T)
    if torch.any(is_normal):
        p_max = p.max(dim=2).values
        L_fp = torch.relu(p_max[is_normal] - fp_margin).mean()
    else:
        L_fp = logits.new_tensor(0.0)

    loss = lambda_and * L_and + lambda_fp * L_fp
    return loss, L_and.detach(), L_fp.detach()

# ====================== Feature builders =====================
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
    scale = np.where(np.abs(sin_th)[...,None] > eps, (theta/(sin_th+eps))[...,None], 1.0)
    w = v * scale
    return np.where((theta < 1e-6)[...,None], v, w)

def _rot_err_vec(R_des, R_act):
    R_rel = np.matmul(np.swapaxes(R_des, -1, -2), R_act)
    return _so3_log(R_rel)

def _rel_log_increment(R):  # (...,T,3,3) -> (...,T,3)
    Tdim = R.shape[-3]
    out = np.zeros(R.shape[:-2] + (3,), dtype=R.dtype)
    if Tdim > 1:
        R_prev = R[..., :-1, :, :]
        R_next = R[..., 1:, :, :]
        R_rel  = np.matmul(np.swapaxes(R_prev, -1, -2), R_next)
        out[..., 1:, :] = _so3_log(R_rel)
    return out

def _time_diff(x):
    d = np.zeros_like(x)
    if x.shape[-3] > 1:
        d[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
    return d

def _flatten_3x4(T):
    return T[..., :3, :4].reshape(*T.shape[:-2], 12)

def build_features_rel_only(d_rel, a_rel):
    """
    입력: d_rel,a_rel : (S,T,L,4,4)  (T_{i-1,i})
    출력: X: (S,T, 42*L)  [des12|act12|p_err|r_err|dp_des|dp_act|dr_des|dr_act] per-link
    """
    S, T, L = d_rel.shape[:3]
    des_12 = _flatten_3x4(d_rel)
    act_12 = _flatten_3x4(a_rel)
    p_des, R_des = d_rel[..., :3, 3], d_rel[..., :3, :3]
    p_act, R_act = a_rel[..., :3, 3], a_rel[..., :3, :3]
    p_err  = p_act - p_des
    r_err  = _rot_err_vec(R_des, R_act)
    dp_des = _time_diff(p_des)
    dp_act = _time_diff(p_act)

    R_des_SK = np.swapaxes(R_des, 1, 2)
    R_act_SK = np.swapaxes(R_act, 1, 2)
    dr_des_SK = _rel_log_increment(R_des_SK)
    dr_act_SK = _rel_log_increment(R_act_SK)
    dr_des = np.swapaxes(dr_des_SK, 1, 2)
    dr_act = np.swapaxes(dr_act_SK, 1, 2)

    feats = np.concatenate([des_12, act_12, p_err, r_err, dp_des, dp_act, dr_des, dr_act], axis=-1)
    S_, T_, L_, _ = feats.shape
    return feats.reshape(S_, T_, L_*42).astype(np.float32)

def build_features_legacy(desired, actual):
    """
    구포맷: desired/actual : (S,T,4,4) -> (S,T,42)
    """
    S, T = desired.shape[:2]
    des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
    act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)
    p_des, R_des = desired[..., :3, 3], desired[..., :3, :3]
    p_act, R_act = actual[...,  :3, 3], actual[...,  :3, :3]
    p_err  = p_act - p_des
    r_err  = _rot_err_vec(R_des, R_act)
    dp_des = _time_diff(p_des)
    dp_act = _time_diff(p_act)
    dr_des = _rel_log_increment(R_des)
    dr_act = _rel_log_increment(R_act)
    X = np.concatenate([des_12, act_12, p_err, r_err, dp_des, dp_act, dr_des, dr_act], axis=2).astype(np.float32)
    return X

# ============================ Train ===========================
if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        link_count = int(input("How many links?: ").strip())
    except Exception:
        link_count = 1
        print("[WARN] Invalid input. Fallback to link_count=1")

    data_path = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")

    # ---- hyperparams ----
    batch_size, lr, wd, seed, epochs = 32, 1e-3, 1e-4, 42, 300
    LAMBDA_FP = 0.05  # 정상 프레임 FP hinge 가중

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("📥 device:", device)
    torch.manual_seed(seed); np.random.seed(seed)

    # ---- Load dataset (auto schema) ----
    d = np.load(data_path, allow_pickle=True)
    keys = set(d.files)
    if {"desired_link_rel","actual_link_rel","desired_link_cum","actual_link_cum","label"}.issubset(keys):
        d_rel = d["desired_link_rel"]   # (S,T,L,4,4)
        a_rel = d["actual_link_rel"]
        labels = d["label"]             # (S,T,M) with M=8*L
        dt = float(d.get("dt", 0.01))
        S, T, L = d_rel.shape[:3]
        M = labels.shape[2]; assert M == 8*L
        # EE(T0N) 제외 → per-link만 사용
        X = build_features_rel_only(d_rel, a_rel)  # (S,T, 42*L)
        y = (1.0 - labels).astype(np.float32)      # 1=fault,0=normal
        print(f"Loaded S={S}, T={T}, L={L}, M={M} | epochs={epochs} | FEAT=42*L={X.shape[2]}")
    elif {"desired","actual","label"}.issubset(keys):
        desired = d["desired"]; actual = d["actual"]; labels = d["label"]
        dt = float(d.get("dt", 0.01))
        S, T = desired.shape[:2]; M = labels.shape[2]
        X = build_features_legacy(desired, actual) # (S,T,42)
        y = (1.0 - labels).astype(np.float32)
        print(f"Loaded LEGACY S={S}, T={T}, M={M} | epochs={epochs} | FEAT=42")
    else:
        raise KeyError(f"Unsupported .npz schema. keys={sorted(keys)}")

    FEAT_DIM = X.shape[2]

    # ---- Dataset & split ----
    full_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_sz = int(0.8 * S); val_sz = S - train_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz], generator=torch.Generator().manual_seed(seed))

    # normalization (train split 기준, torch로 일관)
    X_train = train_ds.dataset.tensors[0][train_ds.indices]  # (train_S, T, FEAT_DIM)
    mu = X_train.reshape(-1, FEAT_DIM).mean(0)               # torch.Tensor (FEAT_DIM,)
    std = X_train.reshape(-1, FEAT_DIM).std(0) + 1e-6

    def norm_tensor(a, mu, std):
        # a: torch (S,T,D); mu/std: torch( D,)
        return (a - mu) / std

    X_all = full_ds.tensors[0]
    y_all = full_ds.tensors[1]
    X_norm = norm_tensor(X_all, mu, std)
    dataset_all = TensorDataset(X_norm, y_all)
    train_ds, val_ds = random_split(dataset_all, [train_sz, val_sz], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # ---- Model / Opt / Sched ----
    model = FaultDiagnosisTransformer(
        input_dim=FEAT_DIM, output_dim=M, max_seq_len=T,
        d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    ckpt_dir = os.path.join(repo_root, "Transformer"); os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"Transformer_link_{link_count}_RELonly.pth")

    best_thr_any = 0.5; best_thr_macro = 0.5; best_thr_exact = 0.5

    # ---- Train loop ----
    for ep in range(1, epochs+1):
        # 커리큘럼: softmin 샤프니스/마진 30ep까지 선형 증가
        beta = 4.0 + 8.0 * min(1.0, ep / 30.0)  # 4→12
        m0   = 0.5 + 1.0 * min(1.0, ep / 30.0)  # 0.5→1.5

        model.train(); train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss, L_and, L_fp = loss_exactall_and_only(
                logits, yb, beta=beta, m0=m0, gamma=2.0,
                lambda_and=1.0, lambda_fp=LAMBDA_FP, fp_margin=0.1
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += loss.item() * xb.size(0)
        tr_loss = train_loss_sum / len(train_ds)

        # ---- Validation ----
        model.eval(); val_loss_sum = 0.0
        probs_col, trues_col = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss, L_and, L_fp = loss_exactall_and_only(
                    logits, yb, beta=beta, m0=m0, gamma=2.0,
                    lambda_and=1.0, lambda_fp=LAMBDA_FP, fp_margin=0.1
                )
                val_loss_sum += loss.item() * xb.size(0)
                p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
                probs_col.append(p.cpu().numpy()); trues_col.append(yb.cpu().numpy())

        val_loss = val_loss_sum / len(val_ds)
        sched.step(val_loss)

        # ---- threshold sweep (진단용) ----
        val_probs = np.concatenate(probs_col, 0)   # (N,T,M)
        val_true  = np.concatenate(trues_col, 0).astype(np.int32)
        ths = np.linspace(0.2, 0.9, 15)

        def any_f1(th):
            pred = (val_probs >= th).astype(np.int32)
            tp=fp=fn=0
            N_,T_,M_ = pred.shape
            for i in range(N_):
                gt_any = (val_true[i]==1).any(axis=1)
                pd_any = (pred[i]==1).any(axis=1)
                tp += np.sum(gt_any & pd_any)
                fp += np.sum((~gt_any) & pd_any)
                fn += np.sum(gt_any & (~pd_any))
            p_ = tp/(tp+fp) if (tp+fp)>0 else 0.0
            r_ = tp/(tp+fn) if (tp+fn)>0 else 0.0
            return 2*p_*r_/(p_+r_) if (p_+r_)>0 else 0.0

        def per_motor_macro_f1(th):
            pred = (val_probs >= th).astype(np.int32)
            tp = ((val_true==1) & (pred==1)).sum(axis=(0,1))
            fp = ((val_true==0) & (pred==1)).sum(axis=(0,1))
            fn = ((val_true==1) & (pred==0)).sum(axis=(0,1))
            prec = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)>0)
            rec  = np.divide(tp, tp+fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)>0)
            f1   = np.divide(2*prec*rec, (prec+rec), out=np.zeros_like(prec), where=(prec+rec)>0)
            return f1.mean()

        def exact_all_acc_at(th):
            pred = (val_probs >= th).astype(np.int32)
            return ((pred == val_true).all(axis=2)).mean()

        f1s_any   = [any_f1(t) for t in ths]
        f1s_macro = [per_motor_macro_f1(t) for t in ths]
        accs_exact= [exact_all_acc_at(t) for t in ths]
        best_thr_any   = float(ths[int(np.argmax(f1s_any))])
        best_thr_macro = float(ths[int(np.argmax(f1s_macro))])
        best_thr_exact = float(ths[int(np.argmax(accs_exact))])

        # 리포트
        exact_at05 = exact_all_acc_at(0.5)
        f1_any_at05 = any_f1(0.5)
        pred05 = (val_probs >= 0.5).astype(np.int32)
        pos_rate = pred05.mean()
        exact_at_best = exact_all_acc_at(best_thr_exact)

        print(f"[{ep:03d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} "
              f"| ExactAllAcc@0.5={exact_at05:.4f} | AnyFault@0.5 F1={f1_any_at05:.4f} "
              f"| best_any_thr={best_thr_any:.2f} | best_macro_thr={best_thr_macro:.2f} | best_exact_thr={best_thr_exact:.2f} "
              f"| pos_rate@0.5={pos_rate:.4f} | ExactAll@best={exact_at_best:.4f}")

    # ---- Save ckpt ----
    torch.save({
        "model_state": model.state_dict(),
        "train_mean": mu.cpu().numpy(), "train_std": std.cpu().numpy(),
        "input_dim": FEAT_DIM, "T": T, "M": M,
        "cfg": dict(d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1,
                    posenc="rope", rope_base=10000.0),
        "label_convention": "1=fault, 0=normal",
        "loss": "AND-softmin + group focal (stop-grad) + FP hinge (no BCE)",
        "loss_params": dict(m0_schedule="0.5->1.5@30ep", gamma=2.0, beta_schedule="4->12@30ep",
                            fp_margin=0.1, lambda_and=1.0, lambda_fp=LAMBDA_FP),
        "features": "REL-only per-link: [des12|act12|p_err|r_err|dp_des|dp_act|dr_des|dr_act] x L",
        "dt": float(dt),
    }, save_path)
    print("✅ saved:", save_path)

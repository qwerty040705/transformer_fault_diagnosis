# Transformer/train_fault_transformer.py
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

# =========================================================
# RoPE (Rotary Positional Embedding) 유틸
# =========================================================
def _build_rope_cache(max_seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=None):
    if device is None: device = torch.device("cpu")
    if dtype is None:  dtype = torch.float32
    idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)  # (d/2,)
    inv_freq = base ** (-idx / head_dim)                             # (d/2,)
    t = torch.arange(max_seq_len, device=device, dtype=dtype)        # (T,)
    freqs = torch.einsum("t,f->tf", t, inv_freq)                     # (T, d/2)
    sin = torch.zeros((max_seq_len, head_dim), device=device, dtype=dtype)
    cos = torch.zeros((max_seq_len, head_dim), device=device, dtype=dtype)
    sin[:, 0::2] = torch.sin(freqs)
    cos[:, 0::2] = torch.cos(freqs)
    sin[:, 1::2] = torch.sin(freqs)
    cos[:, 1::2] = torch.cos(freqs)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, T, 1, d)
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, d)
    return cos, sin

def _rotate_half(x: torch.Tensor):
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (x * cos) + (_rotate_half(x) * sin)

# =========================================================
# Transformer Encoder (RoPE)
# =========================================================
class EncoderBlockRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

        for m in [self.Wq, self.Wk, self.Wv, self.Wo, self.ffn[0], self.ffn[3]]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, rope_cache):
        B, T, D = x.shape
        H, Dh = self.nhead, self.head_dim
        cos, sin = rope_cache

        h = self.ln1(x)
        q = self.Wq(h); k = self.Wk(h); v = self.Wv(h)

        def split_heads(t): return t.view(B, T, H, Dh)
        q = split_heads(q); k = split_heads(k); v = split_heads(v)

        q = apply_rope(q, cos[:, :T, :, :], sin[:, :T, :, :])
        k = apply_rope(k, cos[:, :T, :, :], sin[:, :T, :, :])

        q = q.transpose(1, 2)  # (B,H,T,Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_attn.p if self.training else 0.0
        )  # (B,H,T,Dh)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.Wo(attn_out)

        h2 = self.ln2(x)
        x = x + self.dropout_ffn(self.ffn(h2))
        return x

# =========================================================
# Model with RoPE-based positional encoding
# =========================================================
class FaultDiagnosisTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 42,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: int = 8,
        max_seq_len: int = 1000,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rope_base = rope_base

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EncoderBlockRoPE(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

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

        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def _maybe_build_rope_cache(self, T: int, device, dtype):
        need_build = (
            self.rope_cos.numel() == 0
            or self.rope_cos.shape[1] < T
            or self.rope_cos.device != device
            or self.rope_cos.dtype != dtype
        )
        if need_build:
            cos, sin = _build_rope_cache(
                max_seq_len=T,
                head_dim=self.head_dim,
                base=self.rope_base,
                device=device,
                dtype=dtype,
            )
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if T > self.max_seq_len:
            raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}. Increase max_seq_len.")

        self._maybe_build_rope_cache(T, x.device, x.dtype)
        z = self.input_proj(x) / math.sqrt(self.d_model)
        z = self.pos_drop(z)
        rope_cache = (self.rope_cos, self.rope_sin)
        for blk in self.blocks:
            z = blk(z, rope_cache)
        return self.head(z)

# =========================================================
# ======== Feature helpers (pose/rotation & diffs) ========
# =========================================================
def mat_to_pR(T):  # T: (S,T,4,4) -> (S,T,3), (S,T,3,3)
    p = T[..., :3, 3]
    Rm = T[..., :3, :3]
    return p, Rm

def _vee_skew(A):  # (...,3,3) -> (...,3)
    return np.stack([
        A[..., 2, 1] - A[..., 1, 2],
        A[..., 0, 2] - A[..., 2, 0],
        A[..., 1, 0] - A[..., 0, 1]
    ], axis=-1) / 2.0

def so3_log(Rm):  # vectorized SO(3) log → (...,3)
    tr = np.clip((np.einsum('...ii', Rm) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    A = Rm - np.swapaxes(Rm, -1, -2)
    v = _vee_skew(A)
    sin_th = np.sin(theta)
    eps = 1e-9
    scale = np.where(
        np.abs(sin_th)[..., None] > eps,
        (theta / (sin_th + eps))[..., None],
        np.ones_like(theta)[..., None]
    )
    w = v * scale
    small = (theta < 1e-6)[..., None]
    w = np.where(small, v, w)
    return w

def rotation_error_vec(R_des, R_act):
    R_rel = np.matmul(np.swapaxes(R_des, -1, -2), R_act)
    return so3_log(R_rel)

def rel_log_increment(R):
    S, T, _, _ = R.shape
    out = np.zeros((S, T, 3), dtype=R.dtype)
    if T <= 1:
        return out
    R_prev = R[:, :-1]
    R_next = R[:, 1:]
    R_rel  = np.matmul(np.swapaxes(R_prev, -1, -2), R_next)
    out[:, 1:, :] = so3_log(R_rel)
    return out

def time_diff(x):
    d = np.zeros_like(x)
    d[:, 1:, ...] = x[:, 1:, ...] - x[:, :-1, ...]
    return d

# =========================================================
# Train script
# =========================================================
if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        link_count = int(input("How many links?: ").strip())
    except Exception:
        link_count = 1
        print("[WARN] Invalid input. Fallback to link_count=1")

    data_path   = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")
    batch_size  = 16
    lr, wd      = 1e-3, 1e-4
    seed        = 42
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("📥 device:", device)

    torch.manual_seed(seed); np.random.seed(seed)

    # ─── Load dataset ───────────────────────────────────
    data    = np.load(data_path)
    desired = data["desired"]
    actual  = data["actual"]
    labels  = data["label"]

    S, T, _, _ = desired.shape
    M = labels.shape[2]
    epochs = 100
    print(f"Loaded S={S}, T={T}, M={M} | epochs={epochs}")

    des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
    act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)

    p_des, R_des = mat_to_pR(desired)
    p_act, R_act = mat_to_pR(actual)

    p_err = p_act - p_des
    r_err = rotation_error_vec(R_des, R_act)
    dp_des = time_diff(p_des)
    dp_act = time_diff(p_act)
    dr_des = rel_log_increment(R_des)
    dr_act = rel_log_increment(R_act)

    X = np.concatenate([
        des_12, act_12, p_err, r_err, dp_des, dp_act, dr_des, dr_act
    ], axis=2).astype(np.float32)       # (S,T,42)

    y = (1.0 - labels).astype(np.float32)   # 1=fault, 0=normal

    FEAT_DIM = X.shape[2]

    # ─── Dataset & split ────────────────────────────────
    full_ds   = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_sz  = int(0.8 * S)
    val_sz    = S - train_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz],
                                    generator=torch.Generator().manual_seed(seed))

    X_train = train_ds.dataset.tensors[0][train_ds.indices].numpy()
    μ = X_train.reshape(-1, FEAT_DIM).mean(0)
    σ = X_train.reshape(-1, FEAT_DIM).std(0) + 1e-6

    def _norm_tensor(a: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(((a.numpy() - μ) / σ).astype(np.float32))

    X_norm = _norm_tensor(full_ds.tensors[0]).to(device)
    y_all  = full_ds.tensors[1].to(device)
    dataset_all = TensorDataset(X_norm, y_all)
    train_ds, val_ds = random_split(dataset_all, [train_sz, val_sz],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ─── Model / loss / optim ───────────────────────────
    model = FaultDiagnosisTransformer(
        input_dim=FEAT_DIM,
        output_dim=M,
        max_seq_len=T,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        rope_base=10000.0,
    ).to(device)

    y_np = y.reshape(-1, M)
    pos_counts = y_np.sum(axis=0)
    neg_counts = y_np.shape[0] - pos_counts
    pos_weights = (neg_counts / (pos_counts + 1e-8)).astype(np.float32)
    pos_weight_tensor = torch.from_numpy(pos_weights).to(device)
    print("📊 pos_weight per motor:", pos_weights)

    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    λ_exact = 0.3   # 보조 loss 가중치

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # ─── Train loop ─────────────────────────────────────
    ckpt_dir = os.path.join(repo_root, "Transformer"); os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"Transformer_link_{link_count}.pth")

    for ep in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)

            # BCE + 보조 loss
            bce_loss = bce_loss_fn(logits, yb)
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).float()
            exact_match = (pred == yb).all(dim=2).float()
            exact_loss = 1.0 - exact_match.mean()
            loss = bce_loss + λ_exact * exact_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += loss.item() * xb.size(0)
        tr_loss = train_loss_sum / len(train_ds)

        # ─── Validation ─────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        total_frames, exact_all_frames = 0, 0
        tp = fp = fn = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)

                bce_loss = bce_loss_fn(logits, yb)
                prob = torch.sigmoid(logits)
                pred = (prob >= 0.5).float()
                exact_match = (pred == yb).all(dim=2).float()
                exact_loss = 1.0 - exact_match.mean()
                loss = bce_loss + λ_exact * exact_loss
                val_loss_sum += loss.item() * xb.size(0)

                yb_int = yb.int()
                eq = (pred.int() == yb_int).all(dim=2)
                exact_all_frames += eq.sum().item()
                total_frames += eq.numel()

                gt_fault_any   = (yb_int == 1).any(dim=2)
                pred_fault_any = (pred.int() == 1).any(dim=2)
                tp += ((gt_fault_any & eq).sum().item())
                fp += ((pred_fault_any & (~eq)).sum().item())
                fn += ((gt_fault_any & (~eq)).sum().item())

        val_loss = val_loss_sum / len(val_ds)
        exact_all_acc = exact_all_frames / max(total_frames, 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        print(
            f"[{ep:03d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} "
            f"| Frame-ExactAll-Acc={exact_all_acc:.4f} "
            f"| (Exact-all on fault frames) P={prec:.4f} R={rec:.4f} F1={f1:.4f}"
        )

    torch.save({
        "model_state": model.state_dict(),
        "train_mean" : μ, "train_std": σ,
        "input_dim": FEAT_DIM, "T": T, "M": M,
        "cfg": dict(
            d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1,
            posenc="rope", rope_base=10000.0
        ),
        "label_convention": "1=fault, 0=normal",
        "pos_weight": pos_weights.tolist()
    }, save_path)
    print("✅ saved:", save_path)

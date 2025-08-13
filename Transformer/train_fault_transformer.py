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
    """
    RoPE용 cos/sin 캐시 텐서 생성.
    반환:
      cos: (1, max_seq_len, 1, head_dim)
      sin: (1, max_seq_len, 1, head_dim)
    """
    if device is None: device = torch.device("cpu")
    if dtype is None:  dtype = torch.float32

    # 각 짝(2차원)마다 주파수 스케일
    # theta_i = base^{-2i/d}
    idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)  # (d/2,)
    inv_freq = base ** (-idx / head_dim)                             # (d/2,)

    t = torch.arange(max_seq_len, device=device, dtype=dtype)        # (T,)
    freqs = torch.einsum("t,f->tf", t, inv_freq)                     # (T, d/2)

    # [sin, cos]를 (T, d)로 interleave
    sin = torch.zeros((max_seq_len, head_dim), device=device, dtype=dtype)
    cos = torch.zeros((max_seq_len, head_dim), device=device, dtype=dtype)
    sin[:, 0::2] = torch.sin(freqs)
    cos[:, 0::2] = torch.cos(freqs)
    sin[:, 1::2] = torch.sin(freqs)
    cos[:, 1::2] = torch.cos(freqs)

    # 브로드캐스트 편의를 위해 (1, T, 1, d)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, T, 1, d)
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, d)
    return cos, sin

def _rotate_half(x: torch.Tensor):
    """
    짝수/홀수 채널을 (x_even, x_odd)로 보고 회전 (x_even, x_odd) -> (-x_odd, x_even)
    x: (..., d)
    """
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    # interleave
    x_rot = torch.stack((-x_odd, x_even), dim=-1)  # (..., d/2, 2)
    return x_rot.flatten(-2)                       # (..., d)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    RoPE 적용: x * cos + rotate(x) * sin
    x:   (B, T, H, D)
    cos: (1, T, 1, D)
    sin: (1, T, 1, D)
    """
    return (x * cos) + (_rotate_half(x) * sin)

# =========================================================
# Transformer Encoder (RoPE 버전) — MHA를 직접 구성
# =========================================================
class EncoderBlockRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # QKV, Out proj
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

        # Xavier 초기화
        for m in [self.Wq, self.Wk, self.Wv, self.Wo] + [self.ffn[0], self.ffn[3]]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, rope_cache):
        """
        x: (B, T, d_model)
        rope_cache: (cos, sin) with shapes (1,T,1,D)
        """
        B, T, D = x.shape
        H, Dh = self.nhead, self.head_dim
        cos, sin = rope_cache

        # --- MHA with RoPE ---
        h = self.ln1(x)
        q = self.Wq(h)  # (B,T,D)
        k = self.Wk(h)
        v = self.Wv(h)

        # reshape to multi-head
        def split_heads(t):
            return t.view(B, T, H, Dh)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # RoPE: q, k에만 적용
        q = apply_rope(q, cos[:, :T, :, :], sin[:, :T, :, :])
        k = apply_rope(k, cos[:, :T, :, :], sin[:, :T, :, :])

        # scaled dot-product attention
        q = q.transpose(1, 2)  # (B,H,T,Dh)
        k = k.transpose(1, 2)  # (B,H,T,Dh)
        v = v.transpose(1, 2)  # (B,H,T,Dh)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_attn.p if self.training else 0.0
        )  # (B,H,T,Dh)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
        x = x + self.Wo(attn_out)

        # FFN
        h2 = self.ln2(x)
        x = x + self.dropout_ffn(self.ffn(h2))
        return x

# =========================================================
# Model with RoPE-based positional encoding
# =========================================================
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
        rope_base: float = 10000.0,  # RoPE 주파수 기본값
    ):
        """
        RoPE(상대 위치 r 기반) sin/cos를 사용하는 Transformer.
        - 절대 위치 임베딩을 더하지 않고, q/k에 회전만 적용.
        """
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rope_base = rope_base

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_drop = nn.Dropout(dropout)

        # Encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlockRoPE(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),  # per-timestep output
        )

        # init
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # RoPE 캐시 (디바이스는 입력 때 맞춰서 옮김)
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
        """
        x: (B, T, input_dim)
        out: (B, T, output_dim)
        """
        B, T, _ = x.shape
        if T > self.max_seq_len:
            raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}. "
                             f"Increase max_seq_len when constructing the model.")

        # RoPE 캐시 준비
        self._maybe_build_rope_cache(T, x.device, x.dtype)

        # 입력 프로젝션
        z = self.input_proj(x) / math.sqrt(self.d_model)  # (B,T,d_model)
        z = self.pos_drop(z)

        # Encoder with RoPE
        rope_cache = (self.rope_cos, self.rope_sin)
        for blk in self.blocks:
            z = blk(z, rope_cache)

        return self.head(z)

# =========================================================
# Train script (unchanged logic, but now uses RoPE model)
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
    desired = data["desired"]          # (S,T,4,4)
    actual  = data["actual"]
    labels  = data["label"]            # (S,T,M)

    S, T, _, _ = desired.shape
    M = labels.shape[2]
    epochs = int(0.8 * S)
    print(f"Loaded S={S}, T={T}, M={M} | epochs={epochs}")

    # ─── Build inputs (S,T,24) ──────────────────────────
    des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
    act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)
    X = np.concatenate([des_12, act_12], axis=2).astype(np.float32)   # (S,T,24)
    y = labels.astype(np.float32)                                     # (S,T,M)

    # ─── Dataset & split ────────────────────────────────
    full_ds   = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_sz  = int(0.8 * S)
    val_sz    = S - train_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz],
                                    generator=torch.Generator().manual_seed(seed))

    # ─── Standardize (fit on train only) ────────────────
    X_train = train_ds.dataset.tensors[0][train_ds.indices].numpy()
    μ = X_train.reshape(-1, 24).mean(0)
    σ = X_train.reshape(-1, 24).std(0) + 1e-6

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
        input_dim=24,
        output_dim=M,
        max_seq_len=T,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        rope_base=10000.0,   # 필요시 1e6 등으로 바꿔 긴/짧은 주파수 범위 조정 가능
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # ─── Train loop ─────────────────────────────────────
    ckpt_dir = os.path.join(repo_root, "Transformer"); os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"Transformer_link_{link_count}.pth")

    for ep in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)                  # (B,T,M)
            loss   = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += loss.item() * xb.size(0)
        tr_loss = train_loss_sum / len(train_ds)
        
        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        tp = fp = fn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)  # (B,T,M)
                val_loss_sum += loss_fn(logits, yb).item() * xb.size(0)
                pred = (torch.sigmoid(logits) >= 0.5).int()  # (B,T,M)
                yb_int = yb.int()
                for j in range(xb.size(0)):
                    y_np = yb_int[j].cpu().numpy()
                    p_np = pred[j].cpu().numpy()
                    for m in range(M):
                        if 0 in y_np[:, m]:
                            if 0 in p_np[:, m]:
                                tp += 1
                            else:
                                fn += 1
                        else:
                            if 0 in p_np[:, m]:
                                fp += 1
        val_loss = val_loss_sum / len(val_ds)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * tp / (2 * tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0

        print(f"[{ep:03d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} "
              f"| Det Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")

    # ─── Save ───────────────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "train_mean" : μ, "train_std": σ,
        "input_dim": 24, "T": T, "M": M,
        "cfg": dict(
            d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1,
            posenc="rope", rope_base=10000.0
        )
    }, save_path)
    print("✅ saved:", save_path)

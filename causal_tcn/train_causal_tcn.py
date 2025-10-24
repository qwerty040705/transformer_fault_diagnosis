import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from collections import deque

# ============================================================
#                   Switches / Hyperparams (Updated)
# ============================================================
MODEL_TYPE       = "cnn_transformer"  # "cnn_transformer" | "tcn"(legacy)
PREPROC_MODE     = "cnn1d"            # "cnn1d"(default) | "stft"
WIDER_MODEL      = False
SEED             = 1
EPOCHS           = 30
BATCH_SEQ        = 16
VAL_BS           = 16

# ---------- Real-time evaluation thresholds (more conservative) ----------
EVAL_THETA_H     = 0.60  # hazard prob threshold for FG
EVAL_THETA_M     = 0.85  # ↑ from 0.80 (motor top1 prob threshold for FG)

# Prefix windowing (sampling rebalance)
VIEWS_PER_SEQ    = 3
MIN_WIN          = 32
LOOKBACK_CAP     = 384
ONSET_FOCUS_P        = 0.85
ONSET_JITTER_FRAMES  = 12
LONG_TAIL_P          = 0.12
MIN_POS_FRAC         = 0.75

# ---- Hard negative mining (lightweight replay) ----
HARD_NEG_FRAC        = 0.10   # ~10% of each batch will be "hard negatives" if available
HARD_NEG_CAP         = 2048   # maximum windows stored

# -------------- Loss Weights (ONLY three terms) --------------
LAMBDA_HAZARD          = 0.5
LAMBDA_MOTOR_CE        = 7.0
LAMBDA_TV              = 5e-5

# --------- Motor CE 구성 (updated) ---------
MOTOR_ALPHA_POS        = 0.7
MOTOR_MARGIN_BETA      = 0.05  # ← margin ON
MOTOR_MARGIN_GAMMA     = 0.5
MOTOR_ENTROPY_ETA      = 0.0
LABEL_SMOOTH_EPS       = 0.01  # ← smoothing down from 0.02

# AMP
USE_AMP                = True

# ======== Normalization Switches ========
USE_GLOBAL_ZNORM       = True
USE_CAUSAL_WHITENING   = False
WHITEN_ALPHA_MEAN      = 0.01
WHITEN_ALPHA_VAR       = 0.01

# ======== Regularization & Optim ========
BACKBONE_WEIGHT_DECAY  = 1e-4
HEAD_WEIGHT_DECAY      = 5e-4

# ======== Hazard Head ========
HEAD_TYPE              = "conv_stack"  # "mlp" | "conv1x1" | "conv_stack"
HEAD_DEPTH             = 3
HEAD_HIDDEN            = 256
HEAD_DROPOUT_P         = 0.20
HEAD_CONV_KERNEL       = 11
HEAD_CONV_DILATION     = 1

# ======== Motor Head (variants selectable) ========
# 'perframe_conv' | 'cnn_pool' | 'cnn_rnn' | 'global_pool_mlp' | 'attn_pool'
MOTOR_HEAD_MODE        = "cnn_rnn"  # ← from "cnn_pool"
MOTOR_POOL_COMBINE     = "avgmax"   # used by pooling heads
MOTOR_GRU_HIDDEN       = 256
MOTOR_GRU_LAYERS       = 1
MOTOR_GRU_BIDIR        = False
MOTOR_ATTN_HIDDEN      = 128

# Activations & Norm
ACTIVATION             = "leakyrelu"   # "leakyrelu" | "prelu" | "gelu" | "silu" | "relu"
LEAKY_SLOPE            = 0.10
NORM_TYPE              = "gn"          # "gn" | "ln"
RESIDUAL_SCALE         = 0.1

# LR schedule (One-Cycle)
BACKBONE_BASE_LR       = 5e-5
HEAD_BASE_LR           = 1e-4
ONECYCLE_MAX_LR_MULT   = 1.0
ONECYCLE_PCT_START     = 0.30

# ======== Transformer (경량) ========
TR_D_MODEL             = 128 if not WIDER_MODEL else 192
TR_NHEAD               = 4
TR_DIM_FF              = 1024  # ↑ 256 -> 1024 (8x of 128)
TR_LAYERS              = 3     # ↑ 2 -> 3 (필요시 4로)
TR_DROPOUT             = 0.1
TR_FFN_DEPTH           = 3     # ★ 새로 추가된 깊은 FFN 깊이 (권장 3~4)
TR_FFN_ACT             = "gelu"  # "gelu" | "silu"

# ======== CNN 전처리 ========
# cnn1d 경로
CNN1D_HIDDEN           = 128 if not WIDER_MODEL else 192
CNN1D_BLOCKS           = 2
CNN1D_KERNEL           = 7
CNN1D_DROPOUT          = 0.1

# stft 경로 (causal)
STFT_NFFT              = 64
STFT_WIN               = 64
STFT_HOP               = 8
STFT_TAKE_LOG          = True
STFT_CNN_OUT           = 64
STFT_CNN_BLOCKS        = 2
STFT_CNN_DROPOUT       = 0.1

# ===== Debug/Probe switches =====
PRINT_STATS_EVERY_EPOCH = False
DO_INIT_PROBE           = False
USE_VAL_BATCH_FOR_PROBE = True

# ============================================================
#                      Feature utilities
# ============================================================

def _vee_skew(A: np.ndarray) -> np.ndarray:
    return np.stack([A[...,2,1]-A[...,1,2], A[...,0,2]-A[...,2,0], A[...,1,0]-A[...,0,1]], axis=-1)/2.0

def _so3_log(Rm: np.ndarray) -> np.ndarray:
    tr = np.clip((np.einsum('...ii', Rm)-1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    A = Rm - np.swapaxes(Rm, -1, -2)
    v = _vee_skew(A)
    sin_th = np.sin(theta); eps=1e-9
    scale = np.where(np.abs(sin_th)[...,None]>eps, (theta/(sin_th+eps))[...,None], 1.0)
    w = v*scale
    return np.where((theta<1e-6)[...,None], v, w)

def _time_diff(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x)
    if x.shape[-3] > 1: d[...,1:,:] = x[...,1:,:] - x[..., :-1,:]
    return d

def _twist_from_T(T: np.ndarray) -> np.ndarray:
    Rm = T[..., :3, :3]
    t  = T[..., :3, 3]
    rvec = _so3_log(Rm)
    return np.concatenate([rvec, t], axis=-1)

def build_features_twist_only(d_rel: np.ndarray, a_rel: np.ndarray) -> np.ndarray:
    S, T, L = d_rel.shape[:3]
    tw_des  = _twist_from_T(d_rel)
    tw_act  = _twist_from_T(a_rel)
    tw_err  = tw_act - tw_des
    d_des   = _time_diff(tw_des)
    d_act   = _time_diff(tw_act)
    feats   = np.concatenate([tw_des, tw_act, tw_err, d_des, d_act], axis=-1)  # (S,T,L,30)
    return feats.reshape(S, T, L*30).astype(np.float32)

def build_error_score(d_rel: np.ndarray, a_rel: np.ndarray) -> np.ndarray:
    S, T, L = d_rel.shape[:3]
    return np.zeros((S, T), dtype=np.float32)

# ============================================================
#                Windowing / Collate (prefix-only)
# ============================================================

def _make_windows_onset_biased(x: torch.Tensor, y_bin: torch.Tensor,
                               views_per_seq=3, min_L=32, lookback_cap=384,
                               onset_focus_p=0.6, onset_jitter=20, long_tail_p=0.3):
    T = x.shape[0]; assert T >= min_L
    anypos = (y_bin.max(dim=1).values > 0.5).nonzero(as_tuple=False).squeeze(-1)
    onset  = int(anypos[0].item()) if anypos.numel() > 0 else None

    wins = []
    for _ in range(views_per_seq):
        if onset is not None and random.random() < onset_focus_p:
            t = onset + random.randint(-onset_jitter, onset_jitter)
            t = int(max(min(t, T), min_L))
        else:
            t = random.randint(min_L, T)
        Lmax = t if lookback_cap is None else min(t, lookback_cap)
        L = max(min_L, Lmax) if random.random() < long_tail_p else random.randint(min_L, max(min_L, Lmax))
        t0, t1 = max(0, t-L), t
        wins.append((x[t0:t1], y_bin[t0:t1]))
    return wins

def collate_multi_prefix(batch,
                         views_per_seq=3, min_L=32, lookback_cap=384,
                         onset_focus_p=0.6, onset_jitter=20, long_tail_p=0.3,
                         min_pos_frac=0.5, hard_neg_frac=0.0):
    xs, ys, es, ms = [], [], [], []
    all_wins = []; maxL = 0

    for (x_seq, y_seq, e_seq) in batch:
        ws = _make_windows_onset_biased(
            x_seq, y_seq, views_per_seq, min_L, lookback_cap,
            onset_focus_p, onset_jitter, long_tail_p
        )
        all_wins.extend([(w[0], w[1]) for w in ws])
        for (xw, _) in all_wins[-len(ws):]:
            maxL = max(maxL, xw.shape[0])

    # ensure some positives
    is_pos = [w[1].max().item() > 0.5 for w in all_wins]
    need_pos = int(min_pos_frac*len(all_wins)) - sum(is_pos)
    while need_pos > 0 and len(batch) > 0:
        x_seq, y_seq, _ = random.choice(batch)
        extra = _make_windows_onset_biased(
            x_seq, y_seq, views_per_seq=1, min_L=min_L, lookback_cap=lookback_cap,
            onset_focus_p=1.0, onset_jitter=onset_jitter, long_tail_p=long_tail_p
        )
        all_wins.append((extra[0][0], extra[0][1]))
        need_pos -= 1
        maxL = max(maxL, extra[0][0].shape[0])

    for (xw, yw) in all_wins:
        L, D, M = xw.shape[0], xw.shape[1], yw.shape[1]
        xp = torch.zeros(maxL, D)
        yp = torch.zeros(maxL, M)
        ep = torch.zeros(maxL)
        mp = torch.zeros(maxL)
        xp[:L] = xw
        yp[:L] = yw
        ep[:L] = 0.0
        mp[:L] = 1.0
        xs.append(xp); ys.append(yp); es.append(ep); ms.append(mp)

    return torch.stack(xs,0), torch.stack(ys,0), torch.stack(es,0), torch.stack(ms,0)

def collate_full_sequence(batch):
    maxL = max(x.shape[0] for (x,_,_) in batch)
    xs, ys, es, ms = [], [], [], []
    for (x, y, e) in batch:
        L, D, M = x.shape[0], x.shape[1], y.shape[1]
        xp = torch.zeros(maxL, D); yp = torch.zeros(maxL, M); ep = torch.zeros(maxL); mp = torch.zeros(maxL)
        xp[:L] = x; yp[:L] = y; ep[:L] = 0.0; mp[:L] = 1.0
        xs.append(xp); ys.append(yp); es.append(ep); ms.append(mp)
    return torch.stack(xs,0), torch.stack(ys,0), torch.stack(es,0), torch.stack(ms,0)

# ============================================================
# Label utilities + onset indices
# ============================================================

def ybin_to_class(y_bin: torch.Tensor) -> torch.Tensor:
    is_bg = (y_bin.sum(dim=2) == 0)
    idx = y_bin.argmax(dim=2) + 1
    return idx.masked_fill(is_bg, 0)

def build_any_onset_idx(y_bin, mask):
    y_any = (y_bin.sum(dim=2)>0.5).float()*mask
    B,T = y_any.shape
    has = (y_any>0.5).any(dim=1)
    weights = torch.arange(T, device=y_any.device).view(1,T)+1
    pos = (y_any>0.5).float()*weights
    first = pos.argmax(dim=1)-1
    return torch.where(has, first, torch.full_like(first, -1)).long(), has

# ============================================================
#  Causal whitening (optional)
# ============================================================

def causal_whiten(x: torch.Tensor, mask: torch.Tensor,
                  alpha_mean=0.01, alpha_var=0.01, eps=1e-6):
    maskf = mask.unsqueeze(-1).to(x.dtype)
    cnt   = maskf.cumsum(dim=1).clamp(min=1.0)
    csum  = (x * maskf).cumsum(dim=1)
    csq   = ((x ** 2) * maskf).cumsum(dim=1)
    mean  = csum / cnt
    var   = (csq / cnt) - mean.pow(2)
    std   = (var.clamp_min(0.0) + eps).sqrt()
    out   = (x - mean) / std
    return out * maskf

# ============================================================
# Helpers: Activation & Norm & Init
# ============================================================

def make_act(name: str):
    name = name.lower()
    if name == "leakyrelu":
        return nn.LeakyReLU(LEAKY_SLOPE)
    if name == "prelu":
        return nn.PReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    return nn.ReLU()

class ChannelLayerNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ln = nn.LayerNorm(C)
    def forward(self, x):  # x: [B,C,T]
        return self.ln(x.transpose(1,2)).transpose(1,2)

def make_norm(C: int):
    if NORM_TYPE == "gn":
        gn = nn.GroupNorm(num_groups=1, num_channels=C, affine=True)
        with torch.no_grad():
            gn.weight.fill_(1.0)
            gn.bias.zero_()
        return gn
    else:
        return ChannelLayerNorm(C)

# ------------------ Weight Standardization (+ He gain) ------------------

class WSLinear(nn.Linear):
    """Weight Standardized Linear with variance-preserving gain."""
    def __init__(self, in_features, out_features, bias=True, eps=1e-5):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=LEAKY_SLOPE, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.register_buffer("_ws_gain", torch.tensor(math.sqrt(2.0/(1.0 + LEAKY_SLOPE**2))), persistent=False)

    def forward(self, x):
        w = self.weight
        w = w - w.mean(dim=1, keepdim=True)
        std = w.flatten(1).std(unbiased=False, dim=1, keepdim=True).clamp_min(self.eps)
        w = w / std
        fan_in = w.size(1)
        scale = self._ws_gain / math.sqrt(fan_in)
        return F.linear(x, w * scale, self.bias)

class CausalConv1d(nn.Conv1d):
    """Causal padding + tail cut."""
    def __init__(self, in_ch, out_ch, k, dilation=1):
        padding = (k-1)*dilation
        super().__init__(in_ch, out_ch, k, padding=padding, dilation=dilation)
        self.remove = padding
    def forward(self, x):
        y = super().forward(x)
        return y[:, :, :-self.remove] if self.remove > 0 else y

class ScaledWSConv1d(CausalConv1d):
    """Weight-Standardized Causal Conv1d with variance-preserving gain."""
    def __init__(self, in_ch, out_ch, k, dilation=1, eps=1e-5):
        super().__init__(in_ch, out_ch, k, dilation)
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=LEAKY_SLOPE, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None: nn.init.zeros_(self.bias)
        self.register_buffer("_ws_gain", torch.tensor(math.sqrt(2.0/(1.0 + LEAKY_SLOPE**2))), persistent=False)

    def forward(self, x):
        w = self.weight
        w = w - w.mean(dim=(1,2), keepdim=True)
        std = w.view(w.size(0), -1).std(unbiased=False, dim=1, keepdim=True).clamp_min(self.eps)
        w = w / std.view(-1,1,1)
        fan_in = w.size(1) * w.size(2)
        scale = self._ws_gain / math.sqrt(fan_in)
        w = w * scale
        y = F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y[:, :, :-self.remove] if self.remove > 0 else y

# ===== Stats hook (optional, off by default) =====
class StatHook:
    def __init__(self, name):
        self.name = name
        self.records = []
    @torch.no_grad()
    def __call__(self, module, inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(x): return
        m = x.mean().item()
        s = x.std(unbiased=False).item()
        zr = (x == 0).float().mean().item() if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.GELU, nn.SiLU)) else float('nan')
        self.records.append((self.name, m, s, zr))

def attach_variance_probes(model: nn.Module):
    hooks = []
    def reg(mod, name):
        h = StatHook(name); hooks.append((h, mod.register_forward_hook(h)))
    if hasattr(model, "cnn1d") and isinstance(model.cnn1d, nn.Sequential):
        for i, blk in enumerate(model.cnn1d):
            if isinstance(blk, (nn.Sequential, ScaledWSConv1d, nn.Conv1d, nn.ReLU, nn.LeakyReLU)):
                reg(blk, f"cnn1d[{i}]")
    if hasattr(model, "spec_cnn") and isinstance(model.spec_cnn, nn.Sequential):
        for i, blk in enumerate(model.spec_cnn):
            reg(blk, f"spec_cnn[{i}]")
    if hasattr(model, "transformer_layers"):
        for i, lyr in enumerate(model.transformer_layers):
            reg(lyr, f"tr_lyr[{i}]")
    reg(model, "model_out")
    return hooks

def dump_variance_table(hooks):
    rows = []
    for h, _ in hooks:
        rows += h.records
    print("\n=== Activation/Feature stats (mean, std, zero_rate) ===")
    for name, m, s, zr in rows:
        if math.isnan(zr):
            print(f"{name:<16s} | mean={m:+.4f} | std={s:.4f}")
        else:
            print(f"{name:<16s} | mean={m:+.4f} | std={s:.4f} | zero={zr*100:5.1f}%")

@torch.no_grad()
def probe_activation_stats(model, loader, device, use_causal_whitening=False):
    hooks = attach_variance_probes(model)
    model.eval()
    xb, yb, _, mb = next(iter(loader))
    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
    if use_causal_whitening:
        xb = causal_whiten(xb, mb, alpha_mean=WHITEN_ALPHA_MEAN, alpha_var=WHITEN_ALPHA_VAR)
    haz_logits, motor_logits = model(xb, mb)
    dump_variance_table(hooks)
    print("\n=== Logits stats ===")
    print("hazard logits: mean={:.4f}, std={:.4f}".format(
        haz_logits.mean().item(), haz_logits.std(unbiased=False).item()))
    print("motor  logits: mean={:.4f}, std={:.4f}".format(
        motor_logits.mean().item(), motor_logits.std(unbiased=False).item()))
    for _, h in hooks: h.remove()
    model.train()

# ============================================================
# Masked temporal pooling helpers
# ============================================================

def masked_avg_pool(h_bt_d: torch.Tensor, mask_bt: torch.Tensor):
    m = mask_bt.unsqueeze(-1).to(h_bt_d.dtype)
    s = (h_bt_d * m).sum(dim=1)
    c = m.sum(dim=1).clamp(min=1.0)
    return s / c

def masked_max_pool(h_bt_d: torch.Tensor, mask_bt: torch.Tensor):
    very_neg = torch.finfo(h_bt_d.dtype).min
    m = mask_bt.unsqueeze(-1)
    h = h_bt_d.masked_fill(~m, very_neg)
    return h.max(dim=1).values

def masked_pool(h_bt_d: torch.Tensor, mask_bt: torch.Tensor, how: str = "avgmax"):
    how = how.lower()
    if how == "avg":
        return masked_avg_pool(h_bt_d, mask_bt)
    if how == "max":
        return masked_max_pool(h_bt_d, mask_bt)
    if how == "avgmax":
        a = masked_avg_pool(h_bt_d, mask_bt)
        b = masked_max_pool(h_bt_d, mask_bt)
        return torch.cat([a, b], dim=-1)
    raise ValueError("MOTOR_POOL_COMBINE must be 'avg' | 'max' | 'avgmax'")

def expand_seq_logits_to_time(seq_logits: torch.Tensor, T: int):
    return seq_logits.unsqueeze(1).expand(-1, T, -1)

# ============================================================
# Heads (재사용)
# ============================================================

class PerFrameMLPHead(nn.Module):
    def __init__(self, fin: int, fout: int, hidden: int = 256, depth: int = 3, p: float = 0.2):
        super().__init__()
        layers = []
        d_in = fin
        for _ in range(max(1, depth-1)):
            layers += [WSLinear(d_in, hidden), make_act(ACTIVATION), nn.Dropout(p)]
            d_in = hidden
        out = WSLinear(d_in, fout)
        if out.bias is not None: nn.init.zeros_(out.bias)
        layers += [out]
        self.net = nn.Sequential(*layers)
    def forward(self, h_bt_d, pad_mask=None):
        B, T, D = h_bt_d.shape
        z = self.net(h_bt_d.reshape(B*T, D))
        return z.reshape(B, T, -1)

class PerFrameConvHead(nn.Module):
    def __init__(self, fin, fout, hidden=256, depth=3, k=11, p=0.2, dilation=1):
        super().__init__()
        C_in = fin
        blocks = []
        for _ in range(max(1, depth-1)):
            C_out = hidden
            blocks += [
                make_norm(C_in), make_act(ACTIVATION),
                ScaledWSConv1d(C_in, C_out, k=k, dilation=dilation), nn.Dropout(p)
            ]
            C_in = C_out
        self.blocks = nn.Sequential(*blocks)
        self.proj = ScaledWSConv1d(C_in, fout, k=1, dilation=1)
        if self.proj.bias is not None: nn.init.zeros_(self.proj.bias)

    def forward(self, h_bt_d, pad_mask=None):
        x = h_bt_d.transpose(1,2)          # [B,D,T]
        y = self.blocks(x) if len(self.blocks) > 0 else x
        y = self.proj(y)
        return y.transpose(1,2)            # [B,T,fout]

class CNNPoolingMotorHead(nn.Module):
    def __init__(self, fin, fout, hidden=256, depth=2, k=11, p=0.2, dilation=1, pool_combine="avgmax"):
        super().__init__()
        C_in = fin
        blocks = []
        for _ in range(max(1, depth)):
            C_out = hidden
            blocks += [
                make_norm(C_in), make_act(ACTIVATION),
                ScaledWSConv1d(C_in, C_out, k=k, dilation=dilation), nn.Dropout(p)
            ]
            C_in = C_out
        self.blocks = nn.Sequential(*blocks)
        mult = 2 if pool_combine == "avgmax" else 1
        self.fc = WSLinear(C_in*mult, fout)
        if self.fc.bias is not None: nn.init.zeros_(self.fc.bias)
        self.pool_combine = pool_combine
        self.drop = nn.Dropout(p)

    def forward(self, h_bt_d, pad_mask=None):
        B,T,D = h_bt_d.shape
        x = h_bt_d.transpose(1,2)          # [B,D,T]
        y = self.blocks(x).transpose(1,2)   # [B,T,H]
        if pad_mask is None:
            pad_mask = torch.ones(B,T, dtype=torch.bool, device=h_bt_d.device)
        pooled = masked_pool(y, pad_mask.bool(), self.pool_combine)  # [B,H] or [B,2H]
        pooled = self.drop(pooled)
        logits_seq = self.fc(pooled)
        return expand_seq_logits_to_time(logits_seq, T)

class CNNRNNMotorHead(nn.Module):
    def __init__(self, fin, fout, hidden=256, depth=1, k=5, p=0.2, dilation=1,
                 gru_hidden=256, gru_layers=1, bidir=False):
        super().__init__()
        C_in = fin
        blocks = []
        for _ in range(max(0, depth)):
            C_out = hidden
            blocks += [
                make_norm(C_in), make_act(ACTIVATION),
                ScaledWSConv1d(C_in, C_out, k=k, dilation=dilation), nn.Dropout(p)
            ]
            C_in = C_out
        self.cnn = nn.Sequential(*blocks) if len(blocks)>0 else nn.Identity()
        self.gru = nn.GRU(input_size=C_in, hidden_size=gru_hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=bidir)
        out_dim = gru_hidden * (2 if bidir else 1)
        self.drop = nn.Dropout(p)
        self.proj = WSLinear(out_dim, fout)
        if self.proj.bias is not None: nn.init.zeros_(self.proj.bias)

    def forward(self, h_bt_d, pad_mask=None):
        x = h_bt_d.transpose(1,2)
        y = self.cnn(x).transpose(1,2)
        y = y.float()  # GRU는 float32 필요(MPS/AMP 호환)
        z,_ = self.gru(y)
        z = self.drop(z)
        B,T,H = z.shape
        logits = self.proj(z.reshape(B*T, H)).reshape(B,T,-1)
        return logits

class GlobalPoolMLPMotorHead(nn.Module):
    def __init__(self, fin, fout, hidden=256, depth=2, p=0.2, pool_combine="avgmax"):
        super().__init__()
        mult = 2 if pool_combine == "avgmax" else 1
        layers = []
        d_in = fin*mult
        for _ in range(max(1, depth-1)):
            layers += [WSLinear(d_in, hidden), make_act(ACTIVATION), nn.Dropout(p)]
            d_in = hidden
        out = WSLinear(d_in, fout)
        if out.bias is not None: nn.init.zeros_(out.bias)
        layers += [out]
        self.net = nn.Sequential(*layers)
        self.pool_combine = pool_combine
        self.drop = nn.Dropout(p)

    def forward(self, h_bt_d, pad_mask=None):
        B,T,D = h_bt_d.shape
        if pad_mask is None:
            pad_mask = torch.ones(B,T, dtype=torch.bool, device=h_bt_d.device)
        pooled = masked_pool(h_bt_d, pad_mask.bool(), self.pool_combine)
        pooled = self.drop(pooled)
        logits_seq = self.net(pooled)
        return expand_seq_logits_to_time(logits_seq, T)

class AttnPoolMotorHead(nn.Module):
    def __init__(self, fin, fout, attn_hidden=128, p=0.2):
        super().__init__()
        self.score = nn.Sequential(
            WSLinear(fin, attn_hidden),
            make_act(ACTIVATION),
            WSLinear(attn_hidden, 1)
        )
        self.drop = nn.Dropout(p)
        self.fc = WSLinear(fin, fout)
        if self.fc.bias is not None: nn.init.zeros_(self.fc.bias)

    def forward(self, h_bt_d, pad_mask=None):
        B,T,D = h_bt_d.shape
        if pad_mask is None:
            pad_mask = torch.ones(B,T, dtype=torch.bool, device=h_bt_d.device)
        s = self.score(h_bt_d).squeeze(-1)            # [B,T]
        s = s.masked_fill(~pad_mask.bool(), -1e9)
        a = torch.softmax(s, dim=1)                   # [B,T]
        h = (h_bt_d * a.unsqueeze(-1)).sum(dim=1)     # [B,D]
        h = self.drop(h)
        logits_seq = self.fc(h)                       # [B,M]
        return expand_seq_logits_to_time(logits_seq, T)

# ============================================================
# Transformer (경량, causal) + CNN/STFT 전처리
# ============================================================

class DeepFFN(nn.Module):
    """Deeper feed-forward subnetwork used inside transformer blocks."""
    def __init__(self, d_model, d_ff, depth=3, dropout=0.1, act="gelu"):
        super().__init__()
        act = act.lower()
        Act = nn.GELU if act == "gelu" else nn.SiLU
        layers = []
        for i in range(depth):
            in_d  = d_model if i == 0 else d_ff
            out_d = d_model if i == depth - 1 else d_ff
            layers += [WSLinear(in_d, out_d), Act(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class LiteTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1, ffn_depth=3, ffn_act="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.ff = DeepFFN(d_model, dim_ff, depth=ffn_depth, dropout=dropout, act=ffn_act)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x_bt_d, key_padding_mask=None, causal_mask=None):
        attn_out, _ = self.self_attn(x_bt_d, x_bt_d, x_bt_d,
                                     key_padding_mask=key_padding_mask,
                                     attn_mask=causal_mask)
        x = x_bt_d + self.drop1(attn_out)
        x = self.norm1(x)
        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x

def build_causal_mask(T, device):
    return torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)

class CNN1DPreproc(nn.Module):
    def __init__(self, in_dim, hidden=CNN1D_HIDDEN, blocks=CNN1D_BLOCKS, k=CNN1D_KERNEL, p=CNN1D_DROPOUT):
        super().__init__()
        layers = []
        C_in = in_dim
        for b in range(blocks):
            C_out = hidden
            layers += [
                make_norm(C_in),
                make_act(ACTIVATION),
                ScaledWSConv1d(C_in, C_out, k=k, dilation=1),
                nn.Dropout(p)
            ]
            C_in = C_out
        self.net = nn.Sequential(*layers)
        self.out_dim = C_in

    def forward(self, x_bt_d):
        x = x_bt_d.transpose(1,2)        # [B,D,T]
        y = self.net(x)                  # [B,H,T]
        return y.transpose(1,2)          # [B,T,H]

class STFTSpecPreproc(nn.Module):
    def __init__(self, in_dim, n_fft=STFT_NFFT, win=STFT_WIN, hop=STFT_HOP,
                 take_log=STFT_TAKE_LOG, cnn_out=STFT_CNN_OUT, blocks=STFT_CNN_BLOCKS, p=STFT_CNN_DROPOUT):
        super().__init__()
        self.n_fft, self.win, self.hop = n_fft, win, hop
        self.take_log = take_log
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        chans = [in_dim, cnn_out, cnn_out]
        convs = []
        for i in range(blocks):
            convs += [
                nn.Conv2d(chans[i], chans[i+1], kernel_size=3, padding=1),
                nn.GroupNorm(1, chans[i+1]),
                make_act(ACTIVATION),
                nn.Dropout(p)
            ]
        self.spec_cnn = nn.Sequential(*convs)
        self.out_dim = chans[-1]

    def _stft_mag(self, x_bt_d):
        B,T,D = x_bt_d.shape
        x = x_bt_d.transpose(1,2)                   # [B,D,T]
        mags = []
        for d in range(D):
            xd = x[:, d, :]
            Sd = torch.stft(
                xd, n_fft=self.n_fft, hop_length=self.hop,
                win_length=self.win, window=self.window.to(xd.device),
                center=False,                        # CAUSAL
                return_complex=True
            )
            Md = Sd.abs()
            if self.take_log:
                Md = torch.log1p(Md)
            mags.append(Md.unsqueeze(1))            # [B,1,F,TT]
        spec = torch.cat(mags, dim=1)               # [B,D,F,TT]
        return spec

    def forward(self, x_bt_d):
        B,T,_ = x_bt_d.shape
        spec = self._stft_mag(x_bt_d)               # [B,D,F,TT]
        y = self.spec_cnn(spec)                     # [B,C,F,TT]
        y = F.adaptive_avg_pool2d(y, (1, None)).squeeze(2)   # [B,C,TT]
        y = y.transpose(1,2)                        # [B,TT,C]
        if y.size(1) != T:
            y = F.interpolate(y.transpose(1,2), size=T, mode='linear', align_corners=False).transpose(1,2)
        return y  # [B,T,C]

class CNNTransformerHazardMotor(nn.Module):
    def __init__(self, input_dim, motors):
        super().__init__()
        if PREPROC_MODE == "stft":
            self.pre = STFTSpecPreproc(in_dim=input_dim)
            pre_out = self.pre.out_dim
        else:
            self.cnn1d = CNN1DPreproc(in_dim=input_dim)
            pre_out = self.cnn1d.out_dim

        self.in_proj = WSLinear(pre_out, TR_D_MODEL)

        self.transformer_layers = nn.ModuleList([
            LiteTransformerLayer(
                TR_D_MODEL, TR_NHEAD, TR_DIM_FF, dropout=TR_DROPOUT,
                ffn_depth=TR_FFN_DEPTH, ffn_act=TR_FFN_ACT
            )
            for _ in range(TR_LAYERS)
        ])

        self.feature_dim = TR_D_MODEL

        # Hazard head
        if HEAD_TYPE == "conv1x1":
            self.hazard_head = ScaledWSConv1d(self.feature_dim, 1, 1)
            self._hazard_use_conv = True
        elif HEAD_TYPE == "mlp":
            self.hazard_head = PerFrameMLPHead(fin=self.feature_dim, fout=1,
                                               hidden=HEAD_HIDDEN, depth=HEAD_DEPTH, p=HEAD_DROPOUT_P)
            self._hazard_use_conv = False
        elif HEAD_TYPE == "conv_stack":
            self.hazard_head = PerFrameConvHead(fin=self.feature_dim, fout=1,
                                                hidden=HEAD_HIDDEN, depth=HEAD_DEPTH,
                                                k=HEAD_CONV_KERNEL, p=HEAD_DROPOUT_P, dilation=HEAD_CONV_DILATION)
            self._hazard_use_conv = False
        else:
            raise ValueError("head_type must be 'mlp' | 'conv1x1' | 'conv_stack'")

        # Motor head
        mode = MOTOR_HEAD_MODE.lower()
        if mode == "perframe_conv":
            self.motor_head = PerFrameConvHead(fin=self.feature_dim, fout=motors,
                                               hidden=HEAD_HIDDEN, depth=HEAD_DEPTH,
                                               k=HEAD_CONV_KERNEL, p=HEAD_DROPOUT_P, dilation=HEAD_CONV_DILATION)
            self._motor_needs_mask = False
        elif mode == "cnn_pool":
            self.motor_head = CNNPoolingMotorHead(fin=self.feature_dim, fout=motors,
                                                  hidden=HEAD_HIDDEN, depth=max(1, HEAD_DEPTH-1),
                                                  k=HEAD_CONV_KERNEL, p=HEAD_DROPOUT_P,
                                                  dilation=HEAD_CONV_DILATION,
                                                  pool_combine=MOTOR_POOL_COMBINE)
            self._motor_needs_mask = True
        elif mode == "cnn_rnn":
            self.motor_head = CNNRNNMotorHead(fin=self.feature_dim, fout=motors,
                                              hidden=HEAD_HIDDEN, depth=max(0, HEAD_DEPTH-2),
                                              k=HEAD_CONV_KERNEL, p=HEAD_DROPOUT_P,
                                              dilation=HEAD_CONV_DILATION,
                                              gru_hidden=MOTOR_GRU_HIDDEN, gru_layers=MOTOR_GRU_LAYERS,
                                              bidir=MOTOR_GRU_BIDIR)
            self._motor_needs_mask = False
        elif mode == "global_pool_mlp":
            self.motor_head = GlobalPoolMLPMotorHead(fin=self.feature_dim, fout=motors,
                                                     hidden=HEAD_HIDDEN, depth=HEAD_DEPTH,
                                                     p=HEAD_DROPOUT_P, pool_combine=MOTOR_POOL_COMBINE)
            self._motor_needs_mask = True
        elif mode == "attn_pool":
            self.motor_head = AttnPoolMotorHead(fin=self.feature_dim, fout=motors,
                                                attn_hidden=MOTOR_ATTN_HIDDEN, p=HEAD_DROPOUT_P)
            self._motor_needs_mask = True
        else:
            raise ValueError("MOTOR_HEAD_MODE invalid.")

    def forward(self, x_bt_d, pad_mask=None):
        # Preprocess
        if PREPROC_MODE == "stft":
            h = self.pre(x_bt_d)                  # [B,T,Cp], causal STFT
        else:
            h = self.cnn1d(x_bt_d)                # [B,T,Cp], causal conv

        # Project to d_model
        h = self.in_proj(h)                       # [B,T,Dm]

        # Transformer with causal mask
        B,T,D = h.shape
        device = h.device
        causal_mask = build_causal_mask(T, device)  # bool [T,T], True to mask
        key_pad = None
        if pad_mask is not None:
            key_pad = ~pad_mask.bool()  # True means to ignore

        for lyr in self.transformer_layers:
            h = lyr(h, key_padding_mask=key_pad, causal_mask=causal_mask)

        # Hazard head
        if hasattr(self, "_hazard_use_conv") and self._hazard_use_conv:
            haz = self.hazard_head(h.transpose(1,2)).transpose(1,2)  # [B,T,1]
        else:
            haz = self.hazard_head(h)                                # [B,T,1]

        # Motor head
        if hasattr(self, "_motor_needs_mask") and self._motor_needs_mask:
            mot = self.motor_head(h, pad_mask)                       # [B,T,M]
        else:
            mot = self.motor_head(h)                                 # [B,T,M]

        return haz, mot

# ============================================================
# Losses
# ============================================================

def loss_hazard_bce_with_logits(hazard_logits, y_any, mask, pos_weight=None):
    hz = hazard_logits.squeeze(-1)  # [B,T]
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(hz, y_any, pos_weight=pos_weight, reduction='none')
    else:
        bce = F.binary_cross_entropy_with_logits(hz, y_any, reduction='none')
    bce = bce * mask
    denom = mask.sum().clamp(min=1.0)
    return bce.sum() / denom

def _flatten_pos_frames(motor_logits, y_bin, mask):
    r = (y_bin.sum(dim=2) > 0.5).float()
    pos_mask = (r * mask > 0.5)
    if pos_mask.sum() == 0:
        return None, None, 0
    logits = motor_logits[pos_mask]              # [K,M]
    targets = y_bin[pos_mask].argmax(dim=1)      # [K]
    K = logits.size(0)
    return logits, targets, K

def loss_motor_ce_pos_mean(motor_logits, y_bin, mask, label_smooth=0.0):
    logits, targets, K = _flatten_pos_frames(motor_logits, y_bin, mask)
    if K == 0: return motor_logits.new_zeros(())
    return F.cross_entropy(logits, targets, reduction='mean', label_smoothing=label_smooth)

def loss_motor_ce_macro_seq(motor_logits, y_bin, mask, label_smooth=0.0):
    B, T, M = motor_logits.shape
    r = (y_bin.sum(dim=2) > 0.5).float()
    losses = []
    for b in range(B):
        pos_mask_b = (r[b] * mask[b] > 0.5)
        if pos_mask_b.sum() == 0:
            continue
        logits_b = motor_logits[b][pos_mask_b]         # [K_b, M]
        targets_b = y_bin[b][pos_mask_b].argmax(dim=1) # [K_b]
        ce_b = F.cross_entropy(logits_b, targets_b, reduction='mean', label_smoothing=label_smooth)
        losses.append(ce_b)
    if len(losses) == 0:
        return motor_logits.new_zeros(())
    return torch.stack(losses).mean()

def loss_motor_margin(logits_pos, targets, gamma=0.5):
    if logits_pos is None or logits_pos.numel() == 0:
        dev = targets.device if targets is not None else None
        return torch.tensor(0.0, device=dev, dtype=torch.float32)

    lp32 = logits_pos.float()
    targets = targets.long()

    Bk, C = lp32.shape
    arange = torch.arange(Bk, device=lp32.device)
    tgt_logit = lp32[arange, targets]

    mask = F.one_hot(targets, num_classes=C).bool()
    max_non = lp32.masked_fill(mask, float('-inf')).max(dim=1).values

    margin = F.relu(gamma - (tgt_logit - max_non)).mean()
    return margin.to(logits_pos.dtype)

def loss_motor_entropy(logits_pos, targets=None):
    if logits_pos is None or logits_pos.numel() == 0:
        return torch.tensor(0.0, device=targets.device if targets is not None else None)
    p = F.softmax(logits_pos, dim=-1)
    ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=1).mean()
    return ent

def loss_motor_ce_mixed(motor_logits, y_bin, mask,
                        alpha=0.7, label_smooth=0.0,
                        beta_margin=0.0, margin_gamma=0.5,
                        eta_entropy=0.0):
    L_pos  = loss_motor_ce_pos_mean(motor_logits, y_bin, mask, label_smooth)
    L_macro= loss_motor_ce_macro_seq(motor_logits, y_bin, mask, label_smooth)
    L = alpha * L_pos + (1.0 - alpha) * L_macro
    logits_pos, targets_pos, K = _flatten_pos_frames(motor_logits, y_bin, mask)
    if beta_margin > 0.0:
        L += beta_margin * loss_motor_margin(logits_pos, targets_pos, gamma=margin_gamma)
    if eta_entropy > 0.0:
        L += eta_entropy * loss_motor_entropy(logits_pos, targets_pos)
    return L

def motor_temporal_TV_penalty(motor_logits, mask, lam_tv=1e-4):
    if motor_logits.size(1) < 2 or lam_tv <= 0:
        return motor_logits.new_zeros(())
    p = F.softmax(motor_logits, dim=-1)
    diff = (p[:,1:,:] - p[:,:-1,:]).abs()
    m = (mask[:,1:] * mask[:,:-1]).unsqueeze(-1)
    denom = m.sum() + 1e-9
    tv = (diff * m).sum() / denom
    return lam_tv * tv

# ============================================================
#        Real-time latch (used in eval each epoch)
# ============================================================
@torch.no_grad()
def realtime_latch_dual_fast(
    hazard_logits: torch.Tensor,
    motor_logits: torch.Tensor,
    theta_h: float = 0.50,
    theta_m: float = 0.85,
    kn_k: int = 3,
    kn_nw: int = 8,
    vote_n: int = 5,
    use_dynamic_theta: bool = True,
    high_theta: float = 0.90,
    warmup_frames: int = 40,
):
    device = hazard_logits.device
    B, T, M = motor_logits.shape
    eps = 1e-6
    if hazard_logits.size(-1) == 1:
        q_any = torch.sigmoid(hazard_logits.squeeze(-1)).clamp(eps, 1-eps)  # [B,T]
    else:
        h = hazard_logits.sigmoid().clamp(eps, 1-eps)
        log1mh = torch.log1p(-h)
        logS   = torch.cumsum(log1mh, dim=1)
        logS_any = torch.sum(logS, dim=2)
        S_any = torch.exp(logS_any).clamp(0.0, 1.0)
        q_any = (1.0 - S_any).clamp(0.0, 1.0)

    pi = F.softmax(motor_logits, dim=-1)
    top1_prob, top1_idx = pi.max(dim=-1)

    if use_dynamic_theta:
        theta_vec = torch.full((T,), theta_h, device=device)
        theta_vec[:warmup_frames] = high_theta
        theta_h_all = theta_vec.view(1, T)
    else:
        theta_h_all = torch.full((1, T), theta_h, device=device)

    gate_hits = ((q_any >= theta_h_all) | (top1_prob >= theta_m)).int()
    csum = torch.cumsum(gate_hits, dim=1)
    csum_left = F.pad(csum, (kn_nw, 0))[:, :-kn_nw]
    win_sum = csum - csum_left
    cond = (win_sum >= kn_k)
    t_idx = torch.arange(T, device=device).view(1, T).expand(B, T)
    masked = torch.where(cond, t_idx, torch.full_like(t_idx, T))
    t_star = masked.min(dim=1).values
    has = (t_star != T)

    onehot = F.one_hot(top1_idx, num_classes=M).int()
    csum_oh = torch.cumsum(onehot, dim=1)
    t0 = (t_star - vote_n + 1).clamp(min=0)
    ar_b = torch.arange(B, device=device)
    c_end   = csum_oh[ar_b, t_star.clamp_max(T-1)]
    c_start = torch.where((t0 > 0).unsqueeze(1), csum_oh[ar_b, (t0 - 1).clamp(min=0)], torch.zeros_like(c_end))
    counts = (c_end - c_start).float()
    latched_k = 1 + torch.argmax(counts, dim=1)
    latched_k = torch.where(has, latched_k, torch.zeros_like(latched_k))
    preds = torch.zeros((B, T), dtype=torch.long, device=device)
    time_grid = torch.arange(T, device=device).view(1, T).expand(B, T)
    mask_after = (time_grid >= t_star.view(B, 1)) & has.view(B, 1)
    preds = torch.where(mask_after, latched_k.view(B, 1).expand(B, T), preds)
    return preds

# ============================================================
#                   Hard Negative Memory (replay)
# ============================================================

class HardNegMemory:
    """Stores BG windows that the model falsely gated as FG."""
    def __init__(self, cap=2048):
        self.cap = int(cap)
        self.buf = deque(maxlen=self.cap)

    def push_batch(self, xb, yb, mb, haz_logits, motor_logits):
        with torch.no_grad():
            B = xb.size(0)
            q_any = torch.sigmoid(haz_logits.squeeze(-1))
            pi = F.softmax(motor_logits, dim=-1)
            top1_prob, _ = pi.max(dim=-1)
            gate = (q_any >= EVAL_THETA_H) | (top1_prob >= EVAL_THETA_M)  # [B,T]
            is_bg_win = (yb.sum(dim=(1,2)) <= 0.5)  # no positives in window
            for i in range(B):
                if is_bg_win[i].item() and gate[i].any().item():
                    self.buf.append((
                        xb[i].detach().cpu(), yb[i].detach().cpu(), mb[i].detach().cpu()
                    ))

    def sample(self, k, device):
        """Return up to k windows stacked on device; if empty, return None."""
        k = min(k, len(self.buf))
        if k <= 0:
            return None
        idxs = random.sample(range(len(self.buf)), k)
        xs, ys, ms = [], [], []
        for j in idxs:
            x,y,m = self.buf[j]
            xs.append(x); ys.append(y); ms.append(m)
        xb = torch.stack(xs,0).to(device)
        yb = torch.stack(ys,0).to(device)
        mb = torch.stack(ms,0).to(device)
        return xb, yb, mb

# ============================================================
#                           Main
# ============================================================

def main():
    # Device
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print("Using device:", device)

    # Input
    try:
        link_count = int(input("How many links?: ").strip())
    except Exception:
        link_count = 1; print("Invalid input. Using link_count=1.")

    # ========= Load data: MERGE ALL NPZs IN THE FOLDER =========
    data_dir = os.path.join("data_storage", f"link_{link_count}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    npz_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")])
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    d_rel_list, a_rel_list, labels_list = [], [], []
    T_ref = L_ref = M_ref = None

    for path in npz_files:
        dset = np.load(path, allow_pickle=True)
        d_rel_i  = dset["desired_link_rel"]
        a_rel_i  = dset["actual_link_rel"]
        labels_i = dset["label"]

        S_i, T_i, L_i = d_rel_i.shape[:3]
        M_i = labels_i.shape[2]
        if T_ref is None:
            T_ref, L_ref, M_ref = T_i, L_i, M_i
        else:
            if T_i != T_ref or L_i != L_ref or M_i != M_ref:
                raise ValueError(f"Inconsistent shapes in {path}: "
                                 f"got (T={T_i}, L={L_i}, M={M_i}) vs ref (T={T_ref}, L={L_ref}, M={M_ref})")

        d_rel_list.append(d_rel_i)
        a_rel_list.append(a_rel_i)
        labels_list.append(labels_i)

    d_rel  = np.concatenate(d_rel_list, axis=0)
    a_rel  = np.concatenate(a_rel_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Features & targets
    X = build_features_twist_only(d_rel, a_rel)      # (S,T,L*30)
    err_score_np = build_error_score(d_rel, a_rel)   # (S,T)
    y_bin_np = (1.0 - labels).astype(np.float32)     # (S,T,M)  (fault=1)
    S, T, L = d_rel.shape[:3]
    M = labels.shape[2]
    FEAT = X.shape[2]

    print(f"Loaded {len(npz_files)} files from {data_dir}")
    print(f"Loaded dataset: S={S}, T={T}, L={L}, M={M}, feature_dim={FEAT} (TwistOnly)")

    if (y_bin_np.sum(axis=2) > 1.0).any():
        raise ValueError("Found frames with >1 fault per frame; this script assumes max one.")

    # ===================== Split / Normalize =====================
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    X_all = torch.from_numpy(X).float()
    y_bin = torch.from_numpy(y_bin_np).float()
    e_all = torch.from_numpy(err_score_np).float()

    n_tr  = int(0.8*S); n_va = S - n_tr
    train_ds_i, val_ds_i = random_split(
        TensorDataset(X_all, y_bin, e_all),
        [n_tr, n_va],
        generator=torch.Generator().manual_seed(SEED)
    )
    tr_idx = train_ds_i.indices; va_idx = val_ds_i.indices

    # -------- Input Z-Norm (global, train split) ----------
    if USE_GLOBAL_ZNORM:
        mu  = X_all[tr_idx].reshape(-1, FEAT).mean(dim=0)
        std = X_all[tr_idx].reshape(-1, FEAT).std(dim=0) + 1e-6
        X_all = (X_all - mu)/std
    else:
        mu  = torch.zeros(FEAT); std = torch.ones(FEAT)

    train_ds = TensorDataset(X_all[tr_idx], y_bin[tr_idx], e_all[tr_idx])
    val_ds   = TensorDataset(X_all[va_idx],  y_bin[va_idx],  e_all[va_idx])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SEQ, shuffle=True, drop_last=False,
        collate_fn=lambda b: collate_multi_prefix(
            b, VIEWS_PER_SEQ, MIN_WIN, LOOKBACK_CAP,
            ONSET_FOCUS_P, ONSET_JITTER_FRAMES, LONG_TAIL_P, MIN_POS_FRAC,
            hard_neg_frac=HARD_NEG_FRAC
        )
    )
    val_loader = DataLoader(
        val_ds, batch_size=VAL_BS, shuffle=False, drop_last=False,
        collate_fn=collate_full_sequence
    )

    # ===================== Model =====================
    model = CNNTransformerHazardMotor(input_dim=FEAT, motors=M).to(device)
    save_dir = "CNNTR"

    # ---- Initialize hazard head bias to base any-fault rate ----
    with torch.no_grad():
        y_tr = y_bin[tr_idx]                           # [N,T,M]
        y_any_tr = (y_tr.sum(dim=2) > 0.5).float()     # [N,T]
        p = y_any_tr.mean().clamp(1e-6, 1-1e-6).item()
        logit_p = math.log(p/(1.0-p))
        if HEAD_TYPE in ("conv1x1", "conv_stack"):
            if hasattr(model.hazard_head, 'proj'):
                if model.hazard_head.proj.bias is not None:
                    model.hazard_head.proj.bias.data.fill_(logit_p)
            else:
                if hasattr(model.hazard_head, 'bias') and model.hazard_head.bias is not None:
                    model.hazard_head.bias.data.fill_(logit_p)
        elif HEAD_TYPE == "mlp":
            if hasattr(model.hazard_head.net[-1], 'bias') and model.hazard_head.net[-1].bias is not None:
                model.hazard_head.net[-1].bias.data.fill_(logit_p)

    head_names = ("hazard_head", "motor_head")
    head_params, backbone_params = [], []
    for n,p in model.named_parameters():
        if any(n.startswith(hn) for hn in head_names):
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr":BACKBONE_BASE_LR, "weight_decay":BACKBONE_WEIGHT_DECAY},
        {"params": head_params,     "lr":HEAD_BASE_LR,     "weight_decay":HEAD_WEIGHT_DECAY},
    ])

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[BACKBONE_BASE_LR, HEAD_BASE_LR],
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=ONECYCLE_PCT_START,
        anneal_strategy='cos',
        cycle_momentum=False
    )

    amp_enabled = USE_AMP and (device.type in ("cuda","mps"))
    use_scaler = (device.type == "cuda") and USE_AMP
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # ---- hazard pos_weight (class imbalance) from train set ----
    with torch.no_grad():
        y_any_tr = (y_bin[tr_idx].sum(dim=2) > 0.5).float()
        n_pos = y_any_tr.sum().item()
        n_tot = y_any_tr.numel()
        n_neg = max(n_tot - n_pos, 1.0)
        pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device).float()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"CNNTR_link_{link_count}_{PREPROC_MODE}.pth")

    # ===================== One-batch PROBE (disabled by default) =====================
    if DO_INIT_PROBE:
        hooks = attach_variance_probes(model)
        model.eval()
        with torch.no_grad():
            xb, yb_win, _, mb = next(iter(train_loader))
            xb, yb_win, mb = xb.to(device), yb_win.to(device), mb.to(device)
            if USE_CAUSAL_WHITENING:
                xb = causal_whiten(xb, mb, alpha_mean=WHITEN_ALPHA_MEAN, alpha_var=WHITEN_ALPHA_VAR)
            haz_logits_probe, motor_logits_probe = model(xb, mb)
        dump_variance_table(hooks)
        print("\n=== Logits stats ===")
        print("hazard logits: mean={:.4f}, std={:.4f}".format(haz_logits_probe.mean().item(),
                                                             haz_logits_probe.std(unbiased=False).item()))
        print("motor  logits: mean={:.4f}, std={:.4f}".format(motor_logits_probe.mean().item(),
                                                             motor_logits_probe.std(unbiased=False).item()))
        for _, h in hooks: h.remove()
        model.train()

    # -------- Hard negative memory --------
    hardneg = HardNegMemory(cap=HARD_NEG_CAP)

    # ---- helper: pad/crop to match T ----
    def match_T(x: torch.Tensor, T_target: int) -> torch.Tensor:
        """x: [B,T,...] -> pad with zeros or crop to T_target."""
        T = x.shape[1]
        if T == T_target: return x
        if T < T_target:
            pad_shape = list(x.shape)
            pad_shape[1] = T_target - T
            pad = x.new_zeros(pad_shape)
            return torch.cat([x, pad], dim=1)
        else:
            return x[:, :T_target, ...]

    def compute_metrics(preds, trues, masks):
        correct = ((preds==trues).float()*masks).sum().item()
        total = masks.sum().item()
        acc_raw = correct/(total+1e-9)
        is_bg = (trues==0).float()*masks
        is_fg = (trues>0).float()*masks
        bg_acc = (((preds==0).float()*is_bg).sum().item()) / (is_bg.sum().item()+1e-9)
        fg_rec = (((preds>0).float()*is_fg).sum().item()) / (is_fg.sum().item()+1e-9)

        # FG-ID Acc (only foreground frames, exact motor match)
        fg_mask = (trues>0)&(masks>0.5)
        fg_total = fg_mask.sum().item()+1e-9
        fg_id_acc = ((preds==trues)&fg_mask).sum().item()/fg_total

        # Latency
        def first_onset_idx_from_cls(cls_tensor, mask_tensor):
            B,T = cls_tensor.shape
            has = ((cls_tensor>0) & (mask_tensor>0.5)).any(dim=1)
            weights = torch.arange(T).view(1,T)+1
            pos = ((cls_tensor>0).float()*mask_tensor)*weights
            first = pos.argmax(dim=1)-1
            first = torch.where(has, first, torch.full_like(first, -1))
            return first, has
        gt_onset, gt_has = first_onset_idx_from_cls(trues, masks)
        pr_onset, pr_has = first_onset_idx_from_cls(preds, masks)
        both = gt_has & pr_has
        latency = (pr_onset[both] - gt_onset[both]).float()
        avg_lat = latency.mean().item() if latency.numel()>0 else float('nan')
        return acc_raw, bg_acc, fg_rec, fg_id_acc, avg_lat

    def build_confusion(preds, trues, masks, M):
        """Confusion on FG frames only. Returns matrix [M,M] (true x pred)."""
        with torch.no_grad():
            fg_mask = (trues>0)&(masks>0.5)
            if fg_mask.sum() == 0:
                return torch.zeros(M, M, dtype=torch.long)

            t = trues[fg_mask]-1  # 0..M-1
            p = preds[fg_mask]-1  # 0..M-1

            valid = (t >= 0) & (p >= 0)
            if valid.sum() == 0:
                return torch.zeros(M, M, dtype=torch.long)

            t = t[valid].long()
            p = p[valid].long()

            idx = t*M + p
            binc = torch.bincount(idx, minlength=M*M).reshape(M, M)
            return binc.cpu()

    for ep in range(1, EPOCHS+1):
        # ------------------- Train -------------------
        model.train()
        tr_loss_sum=0.0; tr_steps=0.0
        for xb, yb_win, _, mb in train_loader:
            # 먼저 배치를 device로 이동
            xb, yb_win, mb = xb.to(device), yb_win.to(device), mb.to(device)

            # augment with hard negatives (replay)
            k_hn = int(math.ceil(HARD_NEG_FRAC * xb.size(0)))
            extra = hardneg.sample(k_hn, device)
            if extra is not None:
                xb_h, yb_h, mb_h = extra
                # 현재 배치의 T에 맞추어 패딩/크롭
                T_cur = xb.shape[1]
                xb_h  = match_T(xb_h,  T_cur)
                yb_h  = match_T(yb_h,  T_cur)
                mb_h  = match_T(mb_h,  T_cur)

                xb     = torch.cat([xb_h, xb], dim=0)
                yb_win = torch.cat([yb_h, yb_win], dim=0)
                mb     = torch.cat([mb_h, mb], dim=0)

            if USE_CAUSAL_WHITENING:
                xb = causal_whiten(xb, mb, alpha_mean=WHITEN_ALPHA_MEAN, alpha_var=WHITEN_ALPHA_VAR)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                haz_logits, motor_logits = model(xb, mb)          # [B,T,1], [B,T,M]
                y_any = (yb_win.sum(dim=2) > 0.5).float()         # [B,T]

                L_haz = loss_hazard_bce_with_logits(haz_logits, y_any, mb, pos_weight=pos_weight)
                L_mot = loss_motor_ce_mixed(
                    motor_logits, yb_win, mb,
                    alpha=MOTOR_ALPHA_POS,
                    label_smooth=LABEL_SMOOTH_EPS,
                    beta_margin=MOTOR_MARGIN_BETA,
                    margin_gamma=MOTOR_MARGIN_GAMMA,
                    eta_entropy=MOTOR_ENTROPY_ETA
                )
                L_tv  = motor_temporal_TV_penalty(motor_logits, mb, lam_tv=LAMBDA_TV)
                loss  = LAMBDA_HAZARD*L_haz + LAMBDA_MOTOR_CE*L_mot + L_tv

            if use_scaler:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            scheduler.step()
            tr_loss_sum += loss.item()*mb.sum().item(); tr_steps += mb.sum().item()

            # --- update hard negative memory using current batch logits ---
            try:
                hardneg.push_batch(xb.detach(), yb_win.detach(), mb.detach(),
                                   haz_logits.detach(), motor_logits.detach())
            except Exception:
                pass

        train_loss = tr_loss_sum / max(tr_steps,1.0)

        # ------------------- Val (RealTime only) -------------------
        model.eval()
        val_loss_sum=0.0; val_steps=0.0
        preds_all_LATCH=[]
        trues_all=[]; masks_all=[]
        with torch.no_grad():
            for xb, yb, _, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                if USE_CAUSAL_WHITENING:
                    xb = causal_whiten(xb, mb, alpha_mean=WHITEN_ALPHA_MEAN, alpha_var=WHITEN_ALPHA_VAR)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    haz_logits, motor_logits = model(xb, mb)
                    y_any = (yb.sum(dim=2) > 0.5).float()

                    L_haz = loss_hazard_bce_with_logits(haz_logits, y_any, mb, pos_weight=pos_weight)
                    L_mot = loss_motor_ce_mixed(
                        motor_logits, yb, mb,
                        alpha=MOTOR_ALPHA_POS,
                        label_smooth=LABEL_SMOOTH_EPS,
                        beta_margin=MOTOR_MARGIN_BETA,
                        margin_gamma=MOTOR_MARGIN_GAMMA,
                        eta_entropy=MOTOR_ENTROPY_ETA
                    )
                    L_tv  = motor_temporal_TV_penalty(motor_logits, mb, lam_tv=LAMBDA_TV)
                    vloss  = LAMBDA_HAZARD*L_haz + LAMBDA_MOTOR_CE*L_mot + L_tv

                val_loss_sum += vloss.item()*mb.sum().item(); val_steps += mb.sum().item()

                # ------------ REAL-TIME LATCH 평가 ------------
                pred_cls_LATCH = realtime_latch_dual_fast(
                    haz_logits, motor_logits,
                    theta_h=EVAL_THETA_H, theta_m=EVAL_THETA_M,
                    kn_k=3, kn_nw=8, vote_n=5,
                    use_dynamic_theta=True, high_theta=0.90, warmup_frames=40
                )
                preds_all_LATCH.append(pred_cls_LATCH.cpu())

                trues_all.append(ybin_to_class(yb.cpu()))
                masks_all.append(mb.cpu())

        val_loss = val_loss_sum / max(val_steps,1.0)
        preds_LATCH = torch.cat(preds_all_LATCH,0)
        trues = torch.cat(trues_all,0); masks = torch.cat(masks_all,0)

        acc_raw_L, bg_acc_L, fg_rec_L, fg_id_L, lat_L = compute_metrics(preds_LATCH, trues, masks)

        # ----- Confusion (FG only) -----
        conf = build_confusion(preds_LATCH, trues, masks, M)
        # top-5 confusions excluding diagonal
        conf_np = conf.numpy()
        conf_np_no_diag = conf_np.copy()
        np.fill_diagonal(conf_np_no_diag, 0)
        flat_idx = np.argsort(conf_np_no_diag.ravel())[::-1][:5]
        pairs = [(i//M, i%M, conf_np_no_diag.ravel()[i]) for i in flat_idx if conf_np_no_diag.ravel()[i] > 0]

        # ----- Logs -----
        print(f"[{ep:03d}] [Train] loss={train_loss:.4f} | [Val] loss={val_loss:.4f}")
        print(f"       [RealTime] LatchTop1Acc={acc_raw_L:.4f} | FG-ID Acc={fg_id_L:.4f} "
              f"| BGacc={bg_acc_L:.4f} | FGrec={fg_rec_L:.4f} | Latency {lat_L:.2f}")
        if len(pairs) > 0:
            s_pairs = ", ".join([f"T{t+1}->P{p+1}:{c}" for (t,p,c) in pairs])
            print(f"       [Confusions: top] {s_pairs}")
        else:
            print("       [Confusions: top] (none)")

        # ---- (Optional) per-epoch variance probe ----
        if PRINT_STATS_EVERY_EPOCH and (ep % 1 == 0):
            print("\n[Probe] Activation/variance stats on a fixed {} batch".format(
                "val" if USE_VAL_BATCH_FOR_PROBE else "train"))
            probe_activation_stats(
                model,
                val_loader if USE_VAL_BATCH_FOR_PROBE else train_loader,
                device,
                use_causal_whitening=USE_CAUSAL_WHITENING
            )

    torch.save({
        "model_state_dict": model.state_dict(),
        "train_mean": mu.cpu().numpy(),
        "train_std":  std.cpu().numpy(),
        "input_dim":  FEAT, "num_motors": M,
        "link_count": link_count, "feature_mode": "TwistOnly",
        "model_type": MODEL_TYPE,
        "preproc_mode": PREPROC_MODE,
        "loss_weights": {
            "hazard": LAMBDA_HAZARD,
            "motor_ce": LAMBDA_MOTOR_CE,
            "tv": LAMBDA_TV,
        },
        "motor_loss": {
            "alpha_pos": MOTOR_ALPHA_POS,
            "label_smooth_eps": LABEL_SMOOTH_EPS,
            "margin_beta": MOTOR_MARGIN_BETA,
            "margin_gamma": MOTOR_MARGIN_GAMMA,
            "entropy_eta": MOTOR_ENTROPY_ETA
        },
        "normalization": {
            "use_global_znorm": USE_GLOBAL_ZNORM,
            "use_causal_whitening": USE_CAUSAL_WHITENING,
            "alpha_mean": WHITEN_ALPHA_MEAN, "alpha_var": WHITEN_ALPHA_VAR
        },
        "head": {
            "type": HEAD_TYPE,
            "depth": HEAD_DEPTH,
            "hidden": HEAD_HIDDEN,
            "dropout": HEAD_DROPOUT_P,
            "kernel": HEAD_CONV_KERNEL,
            "dilation": HEAD_CONV_DILATION
        },
        "motor_head": {
            "mode": MOTOR_HEAD_MODE,
            "pool_combine": MOTOR_POOL_COMBINE,
            "gru_hidden": MOTOR_GRU_HIDDEN,
            "gru_layers": MOTOR_GRU_LAYERS,
            "gru_bidir": MOTOR_GRU_BIDIR,
            "attn_hidden": MOTOR_ATTN_HIDDEN
        },
        "transformer": {
            "d_model": TR_D_MODEL, "nhead": TR_NHEAD, "dim_ff": TR_DIM_FF,
            "layers": TR_LAYERS, "dropout": TR_DROPOUT,
            "ffn_depth": TR_FFN_DEPTH, "ffn_act": TR_FFN_ACT
        },
        "stft": {
            "nfft": STFT_NFFT, "win": STFT_WIN, "hop": STFT_HOP, "take_log": STFT_TAKE_LOG
        },
        "eval": {
            "theta_h": EVAL_THETA_H,
            "theta_m": EVAL_THETA_M,
            "vote_n": 5
        },
        "windowing": {
            "views_per_seq": VIEWS_PER_SEQ,
            "min_win": MIN_WIN,
            "lookback_cap": LOOKBACK_CAP,
            "onset_focus_p": ONSET_FOCUS_P,
            "onset_jitter_frames": ONSET_JITTER_FRAMES,
            "long_tail_p": LONG_TAIL_P,
            "min_pos_frac": MIN_POS_FRAC,
            "hard_neg_frac": HARD_NEG_FRAC
        },
        "optim": {
            "backbone_lr": BACKBONE_BASE_LR,
            "head_lr": HEAD_BASE_LR,
            "backbone_wd": BACKBONE_WEIGHT_DECAY,
            "head_wd": HEAD_WEIGHT_DECAY,
            "onecycle_pct_start": ONECYCLE_PCT_START
        },
        "activation": ACTIVATION,
        "leaky_slope": LEAKY_SLOPE,
        "residual_scale": RESIDUAL_SCALE,
        "latch": {
            "theta_h": EVAL_THETA_H, "theta_m": EVAL_THETA_M,
            "K": 3, "Nw": 8, "vote_n": 5,
            "dynamic": True, "high_theta": 0.90, "warmup": 40
        }
    }, save_path)
    print(f"Training complete. Saved final model to {save_path}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
CNN+Transformer Fault Visualization (aligned with training script)

- Loads CNNTransformerHazardMotor with architecture read from checkpoint.
- Uses the same feature builder (TwistOnly) and global Z-Norm as training.
- Runs causal, online inference with realtime_latch_dual_fast (dual gate).
- Renders 3D LASDRA links + 8 motors/link with rotating prop visuals.
- Right panel shows GT fault(=1) and predicted fault probability (per motor).
"""
import sys, os, argparse, math, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d  # 3D→2D projection for always-horizontal labels

# ============================================================
#                    Utils / Seeds / Device
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_ok = getattr(torch.backends, "mps", None)
    if mps_ok is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ============================================================
#                Feature utils (match training)
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
    return np.concatenate([rvec, t], axis=-1)  # (...,6)

def build_features_twist_only(d_rel: np.ndarray, a_rel: np.ndarray) -> np.ndarray:
    S, T, L = d_rel.shape[:3]
    tw_des  = _twist_from_T(d_rel)
    tw_act  = _twist_from_T(a_rel)
    tw_err  = tw_act - tw_des
    d_des   = _time_diff(tw_des)
    d_act   = _time_diff(tw_act)
    feats   = np.concatenate([tw_des, tw_act, tw_err, d_des, d_act], axis=-1)  # (S,T,L,30)
    return feats.reshape(S, T, L*30).astype(np.float32)

# ============================================================
#                Labels and poses from NPZ
# ============================================================

def load_series_from_npz_twist(npz_path: str, seq_idx: int, label_fault_is_one: bool):
    """
    Returns:
        X: [T, D] features (TwistOnly, per training)
        Y_fault: [T, M] where 1 = faulty motor at that frame, 0 otherwise
        meta: dict with desired/actual cumulative poses for 3D and link_count
    """
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    must_have_rel = {"desired_link_rel", "actual_link_rel", "label"}
    must_have_cum = {"desired_link_cum", "actual_link_cum"}
    if not must_have_rel.issubset(keys):
        raise KeyError(f"NPZ missing keys for features: need {sorted(must_have_rel)}, got {sorted(keys)}")
    if not must_have_cum.issubset(keys):
        raise KeyError(f"NPZ missing keys for cum poses: need {sorted(must_have_cum)}, got {sorted(keys)}")

    d_rel = d["desired_link_rel"]  # [S,T,L,4,4]
    a_rel = d["actual_link_rel"]   # [S,T,L,4,4]
    labels = d["label"]            # [S,T,M] (dataset-specific: either fault=1 or health=1)
    Dcum = d["desired_link_cum"]   # [S,T,L,4,4]
    Acum = d["actual_link_cum"]    # [S,T,L,4,4]

    S = d_rel.shape[0]
    assert 0 <= seq_idx < S, f"seq_idx out of range: {seq_idx} not in [0,{S})"

    # Build features as in training
    X_all = build_features_twist_only(d_rel, a_rel)  # [S,T,L*30]
    X = X_all[seq_idx]

    # Convert labels to "fault=1" convention to display/compare
    Y_raw = labels[seq_idx].astype(np.float32)  # [T, M]
    if label_fault_is_one:
        Y_fault = (Y_raw > 0.5).astype(np.float32)
    else:
        # training used y_bin = (1 - labels)
        Y_fault = (1.0 - Y_raw).astype(np.float32)

    meta = {
        "Dcum": Dcum[seq_idx],  # [T, L, 4, 4]
        "Acum": Acum[seq_idx],  # [T, L, 4, 4]
        "link_count": Dcum.shape[2] - 0,
    }
    return X, Y_fault, meta

# ============================================================
#                 Model (same as training)
# ============================================================

# Global switches (activation/norm) — match training defaults
ACTIVATION = "leakyrelu"
NORM_TYPE  = "gn"
LEAKY_SLOPE = 0.10

def make_act(name: str):
    name = name.lower()
    if name == "leakyrelu": return nn.LeakyReLU(LEAKY_SLOPE)
    if name == "prelu":     return nn.PReLU()
    if name == "gelu":      return nn.GELU()
    if name == "silu":      return nn.SiLU()
    return nn.ReLU()

class ChannelLayerNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ln = nn.LayerNorm(C)
    def forward(self, x):  # x: [B,C,T]
        return self.ln(x.transpose(1,2)).transpose(1,2)

def make_norm(C: int):
    if NORM_TYPE == "gn":
        return nn.GroupNorm(num_groups=1, num_channels=C, affine=True)
    else:
        return ChannelLayerNorm(C)

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
        y = F.conv1d(x, w, self.bias, stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)
        return y[:, :, :-self.remove] if self.remove > 0 else y

# ---------------- Heads (reuse from training) ----------------

class WSLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=LEAKY_SLOPE, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None: nn.init.zeros_(self.bias)
        self.register_buffer("_ws_gain", torch.tensor(math.sqrt(2.0/(1.0 + LEAKY_SLOPE**2))), persistent=False)
    def forward(self, x):
        w = self.weight
        w = w - w.mean(dim=1, keepdim=True)
        std = w.flatten(1).std(unbiased=False, dim=1, keepdim=True).clamp_min(self.eps)
        w = w / std
        fan_in = w.size(1)
        scale = self._ws_gain / math.sqrt(fan_in)
        return F.linear(x, w * scale, self.bias)

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
    def masked_avg_pool(self, h_bt_d, mask_bt):
        m = mask_bt.unsqueeze(-1).to(h_bt_d.dtype)
        s = (h_bt_d * m).sum(dim=1)
        c = m.sum(dim=1).clamp(min=1.0)
        return s / c
    def masked_max_pool(self, h_bt_d, mask_bt):
        very_neg = torch.finfo(h_bt_d.dtype).min
        m = mask_bt.unsqueeze(-1)
        h = h_bt_d.masked_fill(~m, very_neg)
        return h.max(dim=1).values
    def masked_pool(self, h_bt_d, mask_bt):
        if self.pool_combine == "avg":
            return self.masked_avg_pool(h_bt_d, mask_bt)
        if self.pool_combine == "max":
            return self.masked_max_pool(h_bt_d, mask_bt)
        a = self.masked_avg_pool(h_bt_d, mask_bt)
        b = self.masked_max_pool(h_bt_d, mask_bt)
        return torch.cat([a, b], dim=-1)
    def forward(self, h_bt_d, pad_mask=None):
        B,T,D = h_bt_d.shape
        x = h_bt_d.transpose(1,2)          # [B,D,T]
        y = self.blocks(x).transpose(1,2)   # [B,T,H]
        if pad_mask is None:
            pad_mask = torch.ones(B,T, dtype=torch.bool, device=h_bt_d.device)
        pooled = self.masked_pool(y, pad_mask.bool())
        pooled = self.drop(pooled)
        logits_seq = self.fc(pooled)
        return logits_seq.unsqueeze(1).expand(-1, T, -1)

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
    def masked_avg_pool(self, h_bt_d, mask_bt):
        m = mask_bt.unsqueeze(-1).to(h_bt_d.dtype)
        s = (h_bt_d * m).sum(dim=1)
        c = m.sum(dim=1).clamp(min=1.0)
        return s / c
    def masked_max_pool(self, h_bt_d, mask_bt):
        very_neg = torch.finfo(h_bt_d.dtype).min
        m = mask_bt.unsqueeze(-1)
        h = h_bt_d.masked_fill(~m, very_neg)
        return h.max(dim=1).values
    def masked_pool(self, h_bt_d, mask_bt):
        if self.pool_combine == "avg": return self.masked_avg_pool(h_bt_d, mask_bt)
        if self.pool_combine == "max": return self.masked_max_pool(h_bt_d, mask_bt)
        a = self.masked_avg_pool(h_bt_d, mask_bt); b = self.masked_max_pool(h_bt_d, mask_bt)
        return torch.cat([a, b], dim=-1)
    def forward(self, h_bt_d, pad_mask=None):
        B,T,D = h_bt_d.shape
        if pad_mask is None:
            pad_mask = torch.ones(B,T, dtype=torch.bool, device=h_bt_d.device)
        pooled = self.masked_pool(h_bt_d, pad_mask.bool())
        pooled = self.drop(pooled)
        logits_seq = self.net(pooled)
        return logits_seq.unsqueeze(1).expand(-1, T, -1)

class AttnPoolMotorHead(nn.Module):
    def __init__(self, fin, fout, attn_hidden=128, p=0.2):
        super().__init__()
        self.score = nn.Sequential(
            WSLinear(fin, attn_hidden), make_act(ACTIVATION), WSLinear(attn_hidden, 1)
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
        return logits_seq.unsqueeze(1).expand(-1, T, -1)

# ---------------- Transformer (+ Preproc) ----------------

class LiteTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            WSLinear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            WSLinear(dim_ff, d_model), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x_bt_d, key_padding_mask=None, causal_mask=None):
        attn_out, _ = self.self_attn(x_bt_d, x_bt_d, x_bt_d,
                                     key_padding_mask=key_padding_mask,
                                     attn_mask=causal_mask)
        x = self.norm1(x_bt_d + self.drop1(attn_out))
        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x

def build_causal_mask(T, device):
    mask = torch.full((T, T), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)  # j>i masked
    return mask

class CNN1DPreproc(nn.Module):
    def __init__(self, in_dim, hidden=128, blocks=2, k=7, p=0.1):
        super().__init__()
        layers = []
        C_in = in_dim
        for _ in range(blocks):
            C_out = hidden
            layers += [
                make_norm(C_in), make_act(ACTIVATION),
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
    def __init__(self, in_dim, n_fft=64, win=64, hop=8,
                 take_log=True, cnn_out=64, blocks=2, p=0.1):
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
    def _stft_mag(self, x_bt_d):  # x:[B,T,D] -> spec:[B, D, F, TT]
        B,T,D = x_bt_d.shape
        x = x_bt_d.transpose(1,2)                   # [B,D,T]
        mags = []
        for d in range(D):
            xd = x[:, d, :]                         # [B,T]
            Sd = torch.stft(xd, n_fft=self.n_fft, hop_length=self.hop,
                             win_length=self.win, window=self.window.to(xd.device),
                             center=True, return_complex=True)
            Md = Sd.abs()                           # [B, F, TT]
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
    def __init__(self, input_dim, motors,
                 preproc_mode="cnn1d",
                 # cnn1d
                 cnn_hidden=128, cnn_blocks=2, cnn_k=7, cnn_dropout=0.1,
                 # stft
                 stft_nfft=64, stft_win=64, stft_hop=8, stft_log=True,
                 stft_cnn_out=64, stft_blocks=2, stft_dropout=0.1,
                 # transformer
                 tr_d_model=128, tr_nhead=4, tr_dim_ff=256, tr_layers=2, tr_dropout=0.1,
                 # heads
                 head_type="conv_stack", head_depth=3, head_hidden=256, head_dropout=0.2,
                 motor_head_mode="cnn_pool", motor_pool="avgmax",
                 motor_gru_hidden=256, motor_gru_layers=1, motor_gru_bidir=False,
                 motor_attn_hidden=128,
                 head_conv_kernel=11, head_conv_dilation=1):
        super().__init__()
        # preproc
        self.preproc_mode = preproc_mode.lower()
        if self.preproc_mode == "stft":
            self.pre = STFTSpecPreproc(
                in_dim=input_dim, n_fft=stft_nfft, win=stft_win, hop=stft_hop,
                take_log=stft_log, cnn_out=stft_cnn_out, blocks=stft_blocks, p=stft_dropout
            )
            pre_out = self.pre.out_dim
        else:
            self.cnn1d = CNN1DPreproc(
                in_dim=input_dim, hidden=cnn_hidden, blocks=cnn_blocks, k=cnn_k, p=cnn_dropout
            )
            pre_out = self.cnn1d.out_dim

        # projection to transformer
        self.in_proj = WSLinear(pre_out, tr_d_model)

        # transformer
        self.transformer_layers = nn.ModuleList([
            LiteTransformerLayer(tr_d_model, tr_nhead, tr_dim_ff, dropout=tr_dropout)
            for _ in range(tr_layers)
        ])
        self.feature_dim = tr_d_model

        # hazard head (per-frame, 1 logit)
        if head_type == "conv1x1":
            self.hazard_head = ScaledWSConv1d(self.feature_dim, 1, 1)
            self._hazard_use_conv = True
        elif head_type == "mlp":
            self.hazard_head = PerFrameMLPHead(fin=self.feature_dim, fout=1,
                                               hidden=head_hidden, depth=head_depth, p=head_dropout)
            self._hazard_use_conv = False
        elif head_type == "conv_stack":
            self.hazard_head = PerFrameConvHead(fin=self.feature_dim, fout=1,
                                                hidden=head_hidden, depth=head_depth,
                                                k=head_conv_kernel, p=head_dropout, dilation=head_conv_dilation)
            self._hazard_use_conv = False
        else:
            raise ValueError("head_type must be 'mlp' | 'conv1x1' | 'conv_stack'")

        # motor head
        mode = motor_head_mode.lower()
        if mode == "perframe_conv":
            self.motor_head = PerFrameConvHead(fin=self.feature_dim, fout=motors,
                                               hidden=head_hidden, depth=head_depth,
                                               k=head_conv_kernel, p=head_dropout, dilation=head_conv_dilation)
            self._motor_needs_mask = False
        elif mode == "cnn_pool":
            self.motor_head = CNNPoolingMotorHead(fin=self.feature_dim, fout=motors,
                                                  hidden=head_hidden, depth=max(1, head_depth-1),
                                                  k=head_conv_kernel, p=head_dropout,
                                                  dilation=head_conv_dilation, pool_combine=motor_pool)
            self._motor_needs_mask = True
        elif mode == "cnn_rnn":
            self.motor_head = CNNRNNMotorHead(fin=self.feature_dim, fout=motors,
                                              hidden=head_hidden, depth=max(0, head_depth-2),
                                              k=head_conv_kernel, p=head_dropout, dilation=head_conv_dilation,
                                              gru_hidden=motor_gru_hidden, gru_layers=motor_gru_layers,
                                              bidir=motor_gru_bidir)
            self._motor_needs_mask = False
        elif mode == "global_pool_mlp":
            self.motor_head = GlobalPoolMLPMotorHead(fin=self.feature_dim, fout=motors,
                                                     hidden=head_hidden, depth=head_depth,
                                                     p=head_dropout, pool_combine=motor_pool)
            self._motor_needs_mask = True
        elif mode == "attn_pool":
            self.motor_head = AttnPoolMotorHead(fin=self.feature_dim, fout=motors,
                                                attn_hidden=motor_attn_hidden, p=head_dropout)
            self._motor_needs_mask = True
        else:
            raise ValueError("motor_head_mode invalid.")

    def forward(self, x_bt_d, pad_mask=None):
        # preproc
        if self.preproc_mode == "stft":
            h = self.pre(x_bt_d)                  # [B,T,Cp]
        else:
            h = self.cnn1d(x_bt_d)                # [B,T,Cp]
        # project
        h = self.in_proj(h)                       # [B,T,Dm]
        # transformer (causal)
        B,T,D = h.shape
        device = h.device
        causal_mask = build_causal_mask(T, device)
        key_pad = None
        if pad_mask is not None:
            key_pad = ~pad_mask.bool()  # True -> mask
        for lyr in self.transformer_layers:
            h = lyr(h, key_padding_mask=key_pad, causal_mask=causal_mask)
        # hazard
        if hasattr(self, "_hazard_use_conv") and self._hazard_use_conv:
            haz = self.hazard_head(h.transpose(1,2)).transpose(1,2)  # [B,T,1]
        else:
            haz = self.hazard_head(h)                                # [B,T,1]
        # motor
        if hasattr(self, "_motor_needs_mask") and self._motor_needs_mask:
            mot = self.motor_head(h, pad_mask)                       # [B,T,M]
        else:
            mot = self.motor_head(h)                                 # [B,T,M]
        return haz, mot

# ============================================================
#             Realtime latch (same as training)
# ============================================================

@torch.no_grad()
def realtime_latch_dual_fast(
    hazard_logits: torch.Tensor,
    motor_logits: torch.Tensor,
    theta_h: float = 0.50,
    theta_m: float = 0.75,
    kn_k: int = 3,
    kn_nw: int = 8,
    vote_n: int = 3,
    use_dynamic_theta: bool = True,
    high_theta: float = 0.90,
    warmup_frames: int = 40,
):
    device = hazard_logits.device
    B, T, M = motor_logits.shape
    eps = 1e-6
    # support hazard=[B,T,1] or [B,T,M]
    if hazard_logits.size(-1) == 1:
        q_any = torch.sigmoid(hazard_logits.squeeze(-1)).clamp(eps, 1-eps)  # [B,T]
    else:
        h = hazard_logits.sigmoid().clamp(eps, 1-eps)  # [B,T,M]
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
#             Arch params from checkpoint (direct)
# ============================================================

def build_model_from_ckpt_meta(meta: dict, input_dim: int, num_motors: int) -> CNNTransformerHazardMotor:
    # Defaults (match training defaults)
    preproc_mode = meta.get("preproc_mode", "cnn1d")

    tr = meta.get("transformer", {})
    tr_d_model = tr.get("d_model", 128)
    tr_nhead   = tr.get("nhead", 4)
    tr_dim_ff  = tr.get("dim_ff", 256)
    tr_layers  = tr.get("layers", 2)
    tr_dropout = tr.get("dropout", 0.1)

    head = meta.get("head", {})
    head_type        = head.get("type", "conv_stack")
    head_depth       = head.get("depth", 3)
    head_hidden      = head.get("hidden", 256)
    head_dropout     = head.get("dropout", 0.2)
    head_conv_kernel = head.get("kernel", 11)
    head_conv_dil    = head.get("dilation", 1)

    mhead = meta.get("motor_head", {})
    motor_head_mode  = mhead.get("mode", "cnn_pool")
    motor_pool       = mhead.get("pool_combine", "avgmax")
    motor_gru_hidden = mhead.get("gru_hidden", 256)
    motor_gru_layers = mhead.get("gru_layers", 1)
    motor_gru_bidir  = mhead.get("gru_bidir", False)
    motor_attn_hidden= mhead.get("attn_hidden", 128)

    stft = meta.get("stft", {})
    stft_nfft = stft.get("nfft", 64)
    stft_win  = stft.get("win", 64)
    stft_hop  = stft.get("hop", 8)
    stft_log  = stft.get("take_log", True)

    # cnn1d defaults
    cnn_hidden = 128
    cnn_blocks = 2
    cnn_k      = 7
    cnn_dropout= 0.1

    return CNNTransformerHazardMotor(
        input_dim=input_dim, motors=num_motors, preproc_mode=preproc_mode,
        cnn_hidden=cnn_hidden, cnn_blocks=cnn_blocks, cnn_k=cnn_k, cnn_dropout=cnn_dropout,
        stft_nfft=stft_nfft, stft_win=stft_win, stft_hop=stft_hop, stft_log=stft_log,
        stft_cnn_out=stft.get("cnn_out", 64), stft_blocks=stft.get("blocks", 2), stft_dropout=stft.get("dropout", 0.1),
        tr_d_model=tr_d_model, tr_nhead=tr_nhead, tr_dim_ff=tr_dim_ff, tr_layers=tr_layers, tr_dropout=tr_dropout,
        head_type=head_type, head_depth=head_depth, head_hidden=head_hidden, head_dropout=head_dropout,
        motor_head_mode=motor_head_mode, motor_pool=motor_pool,
        motor_gru_hidden=motor_gru_hidden, motor_gru_layers=motor_gru_layers, motor_gru_bidir=motor_gru_bidir,
        motor_attn_hidden=motor_attn_hidden,
        head_conv_kernel=head_conv_kernel, head_conv_dilation=head_conv_dil
    )

# ============================================================
#             Streamer: causal online normalization
# ============================================================

class Streamer:
    def __init__(self, model, mu, std, device, latch_cfg=None):
        self.model = model.eval()
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=device)
        self.device = device
        # latch params
        default_latch = dict(theta_h=0.50, theta_m=0.75, K=3, Nw=8, vote_n=3,
                             dynamic=True, high_theta=0.90, warmup=40)
        if latch_cfg is not None:
            default_latch.update(latch_cfg)
        self.latch = default_latch

        self.Xbuf = []  # list of feature vectors
        self.B = 1

    @torch.no_grad()
    def step(self, x_t_np: np.ndarray):
        """
        x_t_np: (D,) numpy
        Returns:
            haz_logits: [1, T, 1]
            motor_probs_last: [M,] softmax at t (last step)
            pred_cls_seq: [T,] latched class over time (1..M, 0=bg)
            pred_onehot_last: [M,] 0/1(only top-1 predicted motor at t)
        """
        self.Xbuf.append(torch.from_numpy(x_t_np).to(self.device).float())
        X = torch.stack(self.Xbuf, dim=0).unsqueeze(0)  # [1,T,D]

        # global z-norm (same stats as training)
        Xn = (X - self.mu) / self.std

        haz, mot = self.model(Xn, None)  # [1,T,1], [1,T,M]

        pred_cls_seq = realtime_latch_dual_fast(
            haz, mot,
            theta_h=self.latch["theta_h"], theta_m=self.latch["theta_m"],
            kn_k=self.latch["K"], kn_nw=self.latch["Nw"], vote_n=self.latch["vote_n"],
            use_dynamic_theta=self.latch["dynamic"],
            high_theta=self.latch["high_theta"], warmup_frames=self.latch["warmup"]
        )[0]  # [T]

        # probs of last step
        pi_last = F.softmax(mot[0, -1], dim=-1)  # [M]
        pred_onehot_last = torch.zeros_like(pi_last)
        pred_onehot_last[torch.argmax(pi_last)] = 1.0

        return haz, pi_last.detach().cpu().numpy(), pred_cls_seq.detach().cpu().numpy(), pred_onehot_last.detach().cpu().numpy()

# ============================================================
#                 3D helpers (motors & blades)
# ============================================================

def prepend_base_identity(cum):
    T, L, _, _ = cum.shape
    I = np.tile(np.eye(4), (T, 1, 1))  # [T,4,4]
    I = I[:, None, :, :]               # [T,1,4,4]
    return np.concatenate([I, cum], axis=1)  # [T,L+1,4,4]

def positions_from_cum(cum):
    return cum[..., :3, 3]

def normalize_by_base(P):
    if P.ndim == 3:
        return P - P[:, :1, :]
    elif P.ndim == 2:
        return P - P[:1, :]
    return P

def _norm(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)

def link_anchor_points(P0, P1, ratio_front: float):
    ratio_back = 1.0 - ratio_front
    p_front = (1.0 - ratio_front) * P0 + ratio_front * P1
    p_back  = (1.0 - ratio_back)  * P0 + ratio_back  * P1
    return p_front, p_back

def cross_four_positions(p_anchor, R_local, arm_len):
    y = R_local[:, 1]
    z = R_local[:, 2]
    return np.array([
        p_anchor + arm_len * y,
        p_anchor - arm_len * y,
        p_anchor + arm_len * z,
        p_anchor - arm_len * z,
    ], dtype=float)  # (4,3)

def orthonormal_blade_axes(n_hat: np.ndarray, R_ref: np.ndarray):
    y_ref = R_ref[:, 1]; z_ref = R_ref[:, 2]
    u = y_ref - np.dot(y_ref, n_hat) * n_hat
    if np.linalg.norm(u) < 1e-6:
        u = z_ref - np.dot(z_ref, n_hat) * n_hat
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0.0, 0.0]) - np.dot([1.0,0.0,0.0], n_hat) * n_hat
    u = _norm(u); v = _norm(np.cross(n_hat, u))
    return u, v

def blade_quad(center, dir_u, dir_v, radius, chord, theta):
    c = np.cos(theta); s = np.sin(theta)
    axis =  c * dir_u + s * dir_v       # blade length direction
    perp = -s * dir_u + c * dir_v       # chord direction
    r_root = 0.25 * radius
    r_tip  = radius
    half_c = 0.5  * chord
    root = center + r_root * axis
    tip  = center + r_tip  * axis
    p1 = root + half_c * perp
    p2 = tip  + half_c * perp
    p3 = tip  - half_c * perp
    p4 = root - half_c * perp
    return np.stack([p1, p2, p3, p4], axis=0)

# ============================================================
#                           Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--seq_idx", type=int, default=0)

    # thresholds/latch
    ap.add_argument("--theta_h", type=float, default=None)
    ap.add_argument("--theta_m", type=float, default=None)
    ap.add_argument("--kn_k", type=int, default=None)
    ap.add_argument("--kn_nw", type=int, default=None)
    ap.add_argument("--vote_n", type=int, default=None)
    ap.add_argument("--dynamic", type=int, default=None)  # 1/0
    ap.add_argument("--high_theta", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)

    # viz / playback
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--data_hz", type=float, default=20.0)
    ap.add_argument("--prepend_base", type=int, default=1)
    ap.add_argument("--fix_origin", type=int, default=1)

    # labels
    ap.add_argument("--label_fault_is_one", type=int, default=1)

    # motors/layout
    ap.add_argument("--motors_per_link", type=int, default=8)
    ap.add_argument("--anchor_ratio", type=float, default=0.85)
    ap.add_argument("--arm_len", type=float, default=0.22)

    # props
    ap.add_argument("--prop_blades", type=int, default=4)
    ap.add_argument("--prop_radius", type=float, default=0.10)
    ap.add_argument("--prop_chord",  type=float, default=0.035)
    ap.add_argument("--prop_alpha",  type=float, default=0.85)
    ap.add_argument("--stem_alpha",  type=float, default=0.95)
    ap.add_argument("--prop_rps", type=float, default=15.0)
    ap.add_argument("--spin_dir_alt", type=int, default=1)

    # video
    ap.add_argument("--save_video", type=int, default=0)
    ap.add_argument("--out", type=str, default="output.mp4")
    ap.add_argument("--video_fps", type=int, default=30)
    ap.add_argument("--codec", type=str, default="libx264")
    ap.add_argument("--bitrate", type=str, default="4000k")
    ap.add_argument("--dpi", type=int, default=150)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- seed/device
    set_seed(args.seed)
    device = pick_device()
    print("📥 device:", device)

    # ---- load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    state = ckpt.get("model_state_dict", None)
    if state is None:
        state = ckpt.get("model_state", ckpt.get("state_dict", None))
        if state is None:
            raise KeyError("Checkpoint missing model state dict (expected 'model_state_dict').")

    D_in  = int(ckpt["input_dim"])
    M_out = int(ckpt["num_motors"])
    mu    = ckpt["train_mean"]
    std   = ckpt["train_std"]

    # model type check (expect cnn_transformer)
    model_type = ckpt.get("model_type", "cnn_transformer")
    if model_type.lower() != "cnn_transformer":
        raise ValueError(f"Checkpoint model_type='{model_type}' != 'cnn_transformer'")

    # latch cfg (defaults + override by eval or args)
    eval_cfg = ckpt.get("eval", {})
    latch_cfg = {
        "theta_h": eval_cfg.get("theta_h", 0.50 if args.theta_h is None else args.theta_h),
        "theta_m": eval_cfg.get("theta_m", 0.75 if args.theta_m is None else args.theta_m),
        "K":  3 if args.kn_k is None else args.kn_k,
        "Nw": 8 if args.kn_nw is None else args.kn_nw,
        "vote_n": 3 if args.vote_n is None else args.vote_n,
        "dynamic": True if args.dynamic is None else bool(args.dynamic),
        "high_theta": 0.90 if args.high_theta is None else args.high_theta,
        "warmup": 40 if args.warmup is None else args.warmup,
    }

    # ---- build model & load (from saved meta)
    model = build_model_from_ckpt_meta(ckpt, D_in, M_out).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"⚠️  load_state_dict non-strict: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:   print("   missing:", missing[:10], "..." if len(missing)>10 else "")
        if unexpected:print("   unexpected:", unexpected[:10], "..." if len(unexpected)>10 else "")
    model.eval()

    # ---- data & features
    X, Y_fault, meta = load_series_from_npz_twist(args.npz, args.seq_idx, bool(args.label_fault_is_one))
    T, D = X.shape
    assert D == D_in, f"Feature dim mismatch: NPZ={D}, ckpt={D_in}"
    M = Y_fault.shape[1]
    assert M == M_out, f"Motor dim mismatch: NPZ label M={M}, ckpt={M_out}"

    Dcum, Acum = meta["Dcum"], meta["Acum"]  # [T, L, 4, 4]
    if args.prepend_base:
        Dcum = prepend_base_identity(Dcum)
        Acum = prepend_base_identity(Acum)
    if Dcum.shape[0] != T:
        raise ValueError(f"Pose length {Dcum.shape[0]} != feature length {T}")

    link_count = Acum.shape[1] - 1  # base prepended
    motors_per_link = args.motors_per_link
    assert motors_per_link * link_count == M_out, f"motors/link mismatch: {motors_per_link}*{link_count} != {M_out}"

    # ---- streamer
    streamer = Streamer(model, mu, std, device, latch_cfg=latch_cfg)

    # ===== Matplotlib layout =====
    rows = link_count
    fig_h = 6 + 1.8 * max(0, rows - 1)
    plt.close("all")
    fig = plt.figure(figsize=(12, fig_h))
    gs = fig.add_gridspec(rows, 2, width_ratios=[3, 2], height_ratios=[1]*rows, wspace=0.35, hspace=0.45)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("LASDRA (desired vs actual)", fontsize = 18)

    axbars = [fig.add_subplot(gs[r, 1]) for r in range(rows)]

    # world limits
    sample = Acum[:min(200, T)]
    p_all = positions_from_cum(sample)
    if args.fix_origin:
        p_all = normalize_by_base(p_all)
    p = p_all.reshape(-1, 3)
    pmin, pmax = p.min(axis=0), p.max(axis=0)
    span = (pmax - pmin).max()
    center = (pmax + pmin)/2
    lim = span*0.8 if span > 0 else 0.5
    ax3d.set_xlim(center[0]-lim, center[0]+lim)
    ax3d.set_ylim(center[1]-lim, center[1]+lim)
    ax3d.set_zlim(center[2]-lim, center[2]+lim)

    desired_lines, actual_lines = [], []
    for _ in range(link_count):
        d_ln, = ax3d.plot([], [], [], linestyle="--", color="g", lw=2.0, alpha=1.0)
        a_ln, = ax3d.plot([], [], [], linestyle="-",  color="k", lw=2.5, alpha=0.45)
        desired_lines.append(d_ln); actual_lines.append(a_ln)

    desired_nodes = ax3d.scatter([], [], [], s=15, c="g", alpha=1.0)
    actual_nodes  = ax3d.scatter([], [], [], s=18, c="k", alpha=0.45)

    base_marker = ax3d.scatter([0], [0], [0], s=120, marker='o',
                               facecolor='k', edgecolor='y', linewidth=2.0, alpha=1.0, zorder=5)
    base_text   = ax3d.text(0.05, 0.05, 0.05, "BASE", color="y", fontsize=10, ha="left", va="bottom")

    assert motors_per_link == 8, "This visual assumes 8 motors per link."
    stems_lines    = [[None]*motors_per_link for _ in range(link_count)]
    blade_patches  = [[[]  for _ in range(motors_per_link)] for _ in range(link_count)]
    fault_texts    = [[None]*motors_per_link for _ in range(link_count)]

    for li in range(link_count):
        for mj in range(motors_per_link):
            ln_stem,  = ax3d.plot([], [], [], color="k", lw=1.2, alpha=args.stem_alpha)
            stems_lines[li][mj] = ln_stem
            patches = []
            for _ in range(args.prop_blades):
                poly = Poly3DCollection([np.zeros((4,3))], closed=True,
                                        facecolor="k", edgecolor="none",
                                        alpha=args.prop_alpha)
                ax3d.add_collection3d(poly)
                patches.append(poly)
            blade_patches[li][mj] = patches
            txt = ax3d.text2D(0, 0, "", color="r", fontsize=8,
                              ha="left", va="center", transform=ax3d.transData)
            fault_texts[li][mj] = txt

    def link_motor_slice(link_idx):
        m0 = link_idx * motors_per_link
        m1 = m0 + motors_per_link
        return m0, m1

    bars_gt_rows, bars_pred_rows, gt_texts_rows = [], [], []
    width = 0.35
    for r, ax in enumerate(axbars):
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Link {r+1} (M1–M{motors_per_link})")
        idxs  = np.arange(motors_per_link)
        x_gt   = idxs - width/2
        x_pred = idxs + width/2

        bars_gt = ax.bar(x_gt, np.zeros(motors_per_link), width=width, alpha=0.35,
                         linewidth=1.0, edgecolor="gray", hatch="//", label="REAL VALUE (fault=1)")
        bars_pd = ax.bar(x_pred, np.zeros(motors_per_link), width=width,
                         label="Pred prob(fault)")

        ax.set_xticks(idxs)
        ax.set_xticklabels([f"M{i+1}" for i in range(motors_per_link)], rotation=0)
        if r == 0: ax.legend(loc="upper right")
        gt_txts = [ax.text(i, 1.02, "", ha="center", va="bottom", fontsize=8) for i in idxs]

        bars_gt_rows.append(bars_gt)
        bars_pred_rows.append(bars_pd)
        gt_texts_rows.append(gt_txts)

    legend_lines = [
        plt.Line2D([0],[0], color="k", lw=2.5, label="Actual Link", alpha=0.45),
        plt.Line2D([0],[0], color="g", lw=2.0, linestyle="--", label="Desired Link"),
        plt.Line2D([0],[0], marker="$x$", color="k", lw=0, markersize=8, label="Motor"),
        plt.Line2D([0],[0], marker="$x$", color="r", lw=0, markersize=8, label="Faulty Motor"),
    ]
    ax3d.legend(handles=legend_lines, loc="upper left")

    status_txt = ax3d.text2D(0.58, 0.92, "", transform=ax3d.transAxes, fontsize=14)

    t_idx = [0]
    prob_last  = np.zeros(M_out, dtype=float)
    predk_last = np.zeros(M_out, dtype=np.uint8)

    prop_phase = 0.0
    dt = 1.0 / max(1e-6, args.data_hz)
    omega = 2.0 * np.pi * args.prop_rps
    if args.spin_dir_alt:
        spin_sign = np.array([1 if (i%2==0) else -1 for i in range(motors_per_link)], dtype=float)
    else:
        spin_sign = np.ones(motors_per_link, dtype=float)

    interval_ms = max(1, int(1000.0 / max(1e-6, args.data_hz * args.speed)))

    def update(_):
        nonlocal prop_phase
        t = t_idx[0]
        if t >= T:
            return []

        # inference step
        _, probs, pred_cls_seq, pred_onehot_last = streamer.step(X[t])
        prob_last[:M_out] = probs[:M_out]

        # latched class at the last time
        k_last = int(pred_cls_seq[-1])
        predk_last[:] = 0
        if k_last > 0:
            predk_last[k_last - 1] = 1

        # GT (fault=1)
        gt_fault = Y_fault[t].astype(float)

        # Poses
        Td = Dcum[t]; Ta = Acum[t]
        P_d = Td[:, :3, 3]
        P_a = Ta[:, :3, 3]
        if args.fix_origin:
            P_d = normalize_by_base(P_d)
            P_a = normalize_by_base(P_a)

        # links
        for i in range(link_count):
            xd, yd, zd = [P_d[i,0], P_d[i+1,0]], [P_d[i,1], P_d[i+1,1]], [P_d[i,2], P_d[i+1,2]]
            desired_lines[i].set_data(xd, yd); desired_lines[i].set_3d_properties(zd)
            xa, ya, za = [P_a[i,0], P_a[i+1,0]], [P_a[i,1], P_a[i+1,1]], [P_a[i,2], P_a[i+1,2]]
            actual_lines[i].set_data(xa, ya);   actual_lines[i].set_3d_properties(za)

        desired_nodes._offsets3d = (P_d[:,0], P_d[:,1], P_d[:,2])
        actual_nodes._offsets3d  = (P_a[:,0], P_a[:,1], P_a[:,2])

        # prop rotation (healthy only)
        prop_phase = (prop_phase + omega * dt) % (2.0*np.pi)

        # motors & labels
        for i in range(link_count):
            R_start = Ta[i,   :3, :3]
            R_end   = Ta[i+1, :3, :3]
            p_front, p_back = link_anchor_points(P_a[i], P_a[i+1], args.anchor_ratio)

            four_front = cross_four_positions(p_front, R_end,   args.arm_len)  # (4,3)
            four_back  = cross_four_positions(p_back,  R_start, args.arm_len)  # (4,3)
            motor_pos  = np.vstack([four_front, four_back])                    # (8,3)

            m0, m1 = link_motor_slice(i)
            R_for_blade = [R_end]*4 + [R_start]*4
            anchors     = [p_front]*4 + [p_back]*4

            for j in range(motors_per_link):
                mi = m0 + j
                is_fault = bool(predk_last[mi])
                color_face = "r" if is_fault else "k"

                pj     = motor_pos[j]
                p_anc  = anchors[j]
                R_ref  = R_for_blade[j]

                # stem
                stems_lines[i][j].set_data([p_anc[0], pj[0]], [p_anc[1], pj[1]])
                stems_lines[i][j].set_3d_properties([p_anc[2], pj[2]])
                stems_lines[i][j].set_color(color_face)

                n_hat = _norm(pj - p_anc)
                u, v  = orthonormal_blade_axes(n_hat, R_ref)

                base_phase = 0.0 if is_fault else (prop_phase * (spin_sign[j]))

                for k, poly in enumerate(blade_patches[i][j]):
                    theta = base_phase + 2.0*np.pi * (k / max(1, args.prop_blades))
                    quad  = blade_quad(pj, u, v, args.prop_radius, args.prop_chord, theta)
                    poly.set_verts([quad])
                    poly.set_facecolor(color_face)
                    poly.set_edgecolor("none")
                    poly.set_alpha(args.prop_alpha)

                # always-horizontal label for faulty motor
                label = fault_texts[i][j]
                if is_fault:
                    x2, y2, _ = proj3d.proj_transform(pj[0], pj[1], pj[2], ax3d.get_proj())
                    label.set_text(f"Link{i+1} Motor{j+1} Fault")
                    label.set_position((x2 + 0.005, y2 + 0.005))
                    label.set_color("r")
                    label.set_alpha(1.0)
                    label.set_visible(True)
                    label.set_transform(ax3d.transData)
                else:
                    label.set_visible(False)

        # right bars
        for r in range(rows):
            m0 = r * motors_per_link
            bars_gt  = bars_gt_rows[r]
            bars_pred= bars_pred_rows[r]
            gt_txts  = gt_texts_rows[r]
            for i_m in range(motors_per_link):
                mi = m0 + i_m
                bars_gt[i_m].set_height(float(gt_fault[mi]))
                bars_pred[i_m].set_height(float(prob_last[mi]))
                is_alarm = bool(predk_last[mi])
                bars_pred[i_m].set_edgecolor("r" if is_alarm else "black")
                bars_pred[i_m].set_linewidth(2.5 if is_alarm else 0.5)
                gt_txts[i_m].set_text("GT:F" if gt_fault[mi] >= 0.5 else "")

        t_real = t / max(1e-6, args.data_hz)
        status_txt.set_text(f"          t = {t_real:4.2f}s")

        t_idx[0] += 1

        artists = desired_lines + actual_lines + [desired_nodes, actual_nodes, base_marker, base_text, status_txt]
        for i in range(link_count):
            artists.extend(stems_lines[i])
            for patches in blade_patches[i]:
                artists.extend(patches)
        for r in range(rows):
            artists.extend(list(bars_gt_rows[r]))
            artists.extend(list(bars_pred_rows[r]))
            artists.extend(gt_texts_rows[r])
        return artists

    # animation
    interval_ms = max(1, int(1000.0 / max(1e-6, args.data_hz * args.speed)))
    ani = FuncAnimation(
        fig, update,
        interval=interval_ms,
        blit=False,
        save_count=T,
        cache_frame_data=False
    )
    plt.tight_layout()

    # save or show
    def _parse_bitrate_to_kbps(b):
        s = str(b).strip().lower()
        if s.endswith("k"): s = s[:-1]
        return int(float(s))

    if args.save_video:
        ext = os.path.splitext(args.out)[1].lower()
        try:
            if ext in [".mp4", ".m4v", ".mov"]:
                writer = FFMpegWriter(fps=args.video_fps, codec=args.codec,
                                      bitrate=_parse_bitrate_to_kbps(args.bitrate))
            elif ext in [".gif"]:
                writer = PillowWriter(fps=args.video_fps)
            else:
                raise ValueError(f"Unsupported extension: {ext} (use .mp4 or .gif)")
            print(f"💾 Saving video to: {args.out}  (fps={args.video_fps}, dpi={args.dpi})")
            ani.save(args.out, writer=writer, dpi=args.dpi)
            print("✅ Done.")
        except Exception as e:
            print(f"❌ Video save failed: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    main()




"""
디버깅용 터미널 입력
python3 visualize_stream_cnntr.py \
  --ckpt CNNTR/CNNTR_link_3_cnn1d.pth \
  --npz  data_storage/link_3/fault_dataset.npz \
  --seq_idx 111 \
  --data_hz 100 --speed 1.0 \
  --prepend_base 1 --fix_origin 1 \
  --label_fault_is_one 1 \
  --motors_per_link 8 \
  --anchor_ratio 0.15 \
  --arm_len 0.15 \
  --prop_blades 4 --prop_radius 0.08 --prop_chord 0.028 --prop_alpha 0.85
"""


""" MP4저장
python3 causal_tcn/visualize_stream_causal_tcn.py \
  --ckpt TCN/TCN_link_1_RELonly_CAUSAL.pth \
  --npz  data_storage/link_1/fault_dataset.npz \
  --seq_idx 100 \
  --threshold 0.5 --kofn 3,5 \
  --data_hz 100 --speed 3.33 \
  --prepend_base 1 --fix_origin 1 \
  --label_fault_is_one 1 \
  --motors_per_link 8 \
  --anchor_ratio 0.15 \
  --arm_len 0.15 \
  --prop_blades 4 --prop_radius 0.08 --prop_chord 0.028 --prop_alpha 0.85 \
  --save_video 1 \
  --out data_storage/link_1/vis.mp4 \
  --video_fps 30 --codec libx264 --bitrate 6000k --dpi 150
"""
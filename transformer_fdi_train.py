#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based LASDRA fault detector (accuracy-first, no topology graph) with:
- OOM-safe: gradient checkpointing, SDPA attention (new API w/ fallback-to-noop), AMP eval/inference_mode
- Per-epoch metrics: Precision/Recall/F1 (macro: all & fault-only), Micro-F1
- Best & Last checkpoint saving
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext

# ============================ Constants & small utils ============================

MOTORS_PER_LINK = 8
EPS = 1e-9

# ------------------------- SO(3) / SE(3) helpers -------------------------

def _unitize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, EPS)
    return vec / norm

def _so3_log_batch(R: np.ndarray) -> np.ndarray:
    """Batched SO(3) log map → vee(skew) * scale, shape (...,3)."""
    trace = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(trace)
    sin_theta = np.sin(theta)
    skew = R - np.swapaxes(R, -1, -2)
    vee = np.stack(
        [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]],
        axis=-1,
    )
    scale = np.empty_like(theta)
    mask = sin_theta > 1e-6
    scale[mask] = theta[mask] / (2.0 * sin_theta[mask])
    theta_small = theta[~mask]
    scale[~mask] = 0.5 + (theta_small * theta_small) / 12.0
    return vee * scale[..., None]

def _axis_from_R(R: np.ndarray) -> np.ndarray:
    """Axis on S^2 from rotation matrix via log(R)."""
    w = _so3_log_batch(R)
    return _unitize(w).astype(np.float32, copy=False)

def compute_se3_residual(desired: np.ndarray, actual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    desired, actual: (..., 4,4)
    return:
      omega_resid: (...,3)  log(R_d^T R_a)
      pos_err    : (...,3)  p_a - p_d
    """
    pos_err = (actual[..., :3, 3] - desired[..., :3, 3]).astype(np.float32, copy=False)
    rot_err = np.matmul(np.swapaxes(desired[..., :3, :3], -1, -2), actual[..., :3, :3])
    omega = _so3_log_batch(rot_err).astype(np.float32, copy=False)
    return omega, pos_err

# ------------------------- Motor geometry (optional features) -------------------------

def default_motor_layout(link_count: int) -> np.ndarray:
    """Canonical layout of 8 motors per link (unitized), shape (L,8,3)."""
    base = np.array(
        [
            [ 1.0,  0.0, 0.5], [-1.0,  0.0, 0.5],
            [ 0.0,  1.0, 0.5], [ 0.0, -1.0, 0.5],
            [ 0.5,  0.5, 1.0], [-0.5,  0.5, 1.0],
            [ 0.5, -0.5, 1.0], [-0.5, -0.5, 1.0],
        ],
        dtype=np.float32,
    )
    layout = np.tile(_unitize(base)[None, :, :], (link_count, 1, 1))
    return layout

def compute_motor_directions(rotations: np.ndarray, motor_layout: np.ndarray) -> np.ndarray:
    """rotations: (S,T,L,3,3), motor_layout: (L,8,3) → (S,T,L,8,3) world-frame unit dirs"""
    dirs = np.einsum("stlij,lmj->stlmi", rotations, motor_layout, optimize=True)
    return _unitize(dirs.astype(np.float32, copy=False))

# ------------------------- Labels collapsing -------------------------

def build_link_targets(label_matrix: np.ndarray) -> np.ndarray:
    """
    label_matrix: (S,T, 8*L) with 1=healthy, 0=faulty.
    Collapse to per-(S,T) categorical link index: 0=healthy, 1..L faulty link id.
    """
    S, T, M = label_matrix.shape
    if M % MOTORS_PER_LINK != 0:
        raise ValueError(f"Label width {M} not divisible by {MOTORS_PER_LINK}.")
    L = M // MOTORS_PER_LINK
    reshaped = label_matrix.reshape(S, T, L, MOTORS_PER_LINK)
    fault_flags = reshaped.min(axis=-1) < 1  # (S,T,L)
    targets = fault_flags.argmax(axis=-1) + 1
    healthy = ~fault_flags.any(axis=-1)
    targets[healthy] = 0
    return targets.astype(np.int64, copy=False)

def build_motor_targets(label_matrix: np.ndarray) -> np.ndarray:
    """
    label_matrix: (S,T, 8*L) with 1=healthy, 0=faulty.
    Collapse to global motor categorical: 0=healthy, 1..(8L) faulty motor id.
    """
    S, T, M = label_matrix.shape
    faults = label_matrix < 1
    has_fault = faults.any(axis=-1)
    indices = np.argmax(faults, axis=-1) + 1
    indices[~has_fault] = 0
    return indices.astype(np.int64, copy=False)

# ============================ Feature pack ============================

@dataclass
class FeaturePack:
    node_features: np.ndarray  # (S,T,L,F_node)
    link_labels:   np.ndarray  # (S,T)
    motor_labels:  np.ndarray  # (S,T)
    node_dim: int
    link_count: int

# ============================ Feature builder ============================

def build_features(
    desired_cum: np.ndarray,
    actual_cum:  np.ndarray,
    labels:      np.ndarray,
    baseline_window: int = 120,
    include_motor_features: bool = True,
    component_threshold_deg: float = 18.0,
) -> FeaturePack:
    S, T, L, _, _ = desired_cum.shape

    # Residuals & energy
    omega_resid, pos_err = compute_se3_residual(desired_cum, actual_cum)  # (S,T,L,3)
    energy = (omega_resid**2 + pos_err**2).sum(axis=-1, keepdims=True).astype(np.float32)

    # Absolute rotation/axis & temporal angular velocity
    R_act = actual_cum[..., :3, :3]
    axis_i = _axis_from_R(R_act)                                     # (S,T,L,3)
    omega_abs = np.zeros((S, T, L, 3), dtype=np.float32)
    if T > 1:
        Rt = np.matmul(np.swapaxes(R_act[:, :-1], -1, -2), R_act[:, 1:])
        omega_abs[:, :-1] = _so3_log_batch(Rt).astype(np.float32, copy=False)

    # Motor-derived features (optional)
    if include_motor_features:
        layout = default_motor_layout(L)
        m_now = compute_motor_directions(R_act, layout)
        base_window = max(1, min(baseline_window, T))
        m_base = _unitize(m_now[:, :base_window].mean(axis=1, keepdims=False))  # (S,L,8,3)
        dots = np.clip((m_now * m_base[:, None]).sum(axis=-1), -1.0, 1.0)       # (S,T,L,8)
        ang  = np.arccos(dots)[..., None]                                       # (S,T,L,8,1)
        diff = (m_now - m_base[:, None])                                        # (S,T,L,8,3)
        per_motor_flat = np.concatenate([m_now, diff, ang], axis=-1)            # (S,T,L,8,7)
        per_motor_flat = per_motor_flat.reshape(S, T, L, MOTORS_PER_LINK * 7).astype(np.float32, copy=False)

        cos_thr = math.cos(math.radians(component_threshold_deg))
        cos = np.einsum("stlmk,stlpk->stlmp", m_now, m_now, optimize=True)      # (S,T,L,8,8)
        upper = np.triu(np.ones((MOTORS_PER_LINK, MOTORS_PER_LINK), dtype=bool), k=1)
        edges = (cos >= cos_thr)[..., upper].sum(axis=-1)                        # (S,T,L)
        denom = MOTORS_PER_LINK * (MOTORS_PER_LINK - 1) / 2.0
        density = (edges / denom).astype(np.float32, copy=False)[..., None]      # (S,T,L,1)
    else:
        per_motor_flat = np.zeros((S, T, L, 0), dtype=np.float32)
        density = np.zeros((S, T, L, 1), dtype=np.float32)

    node_parts = [
        omega_resid.astype(np.float32, copy=False),
        pos_err.astype(np.float32, copy=False),
        energy,
        axis_i,
        omega_abs,
        density,
    ]
    if per_motor_flat.shape[-1] > 0:
        node_parts.append(per_motor_flat)

    node_features = np.concatenate(node_parts, axis=-1)  # (S,T,L,F)
    link_labels  = build_link_targets(labels)
    motor_labels = build_motor_targets(labels)

    return FeaturePack(
        node_features=node_features,
        link_labels=link_labels,
        motor_labels=motor_labels,
        node_dim=node_features.shape[-1],
        link_count=node_features.shape[2],
    )

# ============================ Shard discovery & loader ============================

def discover_shards(root: str) -> List[Path]:
    pattern = str(Path(root) / "fault_dataset_shard_*.npz")
    return [Path(p) for p in sorted(glob.glob(pattern))]

def load_all_shards(
    paths: Sequence[Path],
    baseline_window: int,
    include_motor_features: bool,
    component_threshold_deg: float,
) -> FeaturePack:
    node_buf: List[np.ndarray] = []
    link_buf: List[np.ndarray] = []
    motor_buf: List[np.ndarray] = []
    node_dim = None
    link_count = None

    total = len(paths)
    for idx, path in enumerate(paths, 1):
        with np.load(path) as data:
            desired = data["desired_link_cum"]
            actual  = data["actual_link_cum"]
            labels  = data["label"]

        pack = build_features(
            desired_cum=desired, actual_cum=actual, labels=labels,
            baseline_window=baseline_window,
            include_motor_features=include_motor_features,
            component_threshold_deg=component_threshold_deg,
        )
        node_buf.append(pack.node_features)
        link_buf.append(pack.link_labels)
        motor_buf.append(pack.motor_labels)

        if node_dim is None:
            node_dim = pack.node_dim
            link_count = pack.link_count
        else:
            if node_dim != pack.node_dim:
                raise ValueError("Inconsistent feature dim across shards.")
            if link_count != pack.link_count:
                raise ValueError("Inconsistent link count across shards.")

        print(f"[data] processed shard {idx}/{total}", flush=True)

    node_features = np.concatenate(node_buf, axis=0)
    link_labels   = np.concatenate(link_buf,  axis=0)
    motor_labels  = np.concatenate(motor_buf, axis=0)

    return FeaturePack(
        node_features=node_features,
        link_labels=link_labels,
        motor_labels=motor_labels,
        node_dim=node_dim or 0,
        link_count=link_count or 0,
    )

# ============================ Dataset & normalization ============================

class FeatureWindowDataset(Dataset):
    """
    Returns causal windows:
      - node_window: (T, L, F_node)
      - link_label:  int (last time step)
      - motor_label: int (last time step)
    """
    def __init__(
        self,
        node_features: np.ndarray,
        link_labels:   np.ndarray,
        motor_labels:  np.ndarray,
        window: int,
        pos_stride: int,
        neg_stride: int,
    ) -> None:
        if node_features.shape[:2] != link_labels.shape[:2]:
            raise ValueError("node_features and link_labels shape mismatch.")
        if window <= 0 or window > node_features.shape[1]:
            raise ValueError("invalid window length.")

        self.node = node_features
        self.link_full = link_labels
        self.motor_full = motor_labels
        self.window = window

        N, T, L, _ = node_features.shape
        seq_idx: List[int] = []
        start_idx: List[int] = []
        ys_link: List[int] = []
        ys_motor: List[int] = []

        for seq in range(N):
            end = window
            while end <= T:
                start = end - window
                y_link = int(link_labels[seq, end - 1])
                y_motor = int(motor_labels[seq, end - 1])
                seq_idx.append(seq)
                start_idx.append(start)
                ys_link.append(y_link)
                ys_motor.append(y_motor)
                step = pos_stride if y_link != 0 else neg_stride
                end += step

        self.seq_idx = np.asarray(seq_idx, dtype=np.int32)
        self.start_idx = np.asarray(start_idx, dtype=np.int32)
        self.link_labels = np.asarray(ys_link, dtype=np.int64)
        self.motor_labels = np.asarray(ys_motor, dtype=np.int64)

    def __len__(self) -> int:
        return self.link_labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.seq_idx[idx]
        start = self.start_idx[idx]
        end = start + self.window
        node_w = self.node[seq, start:end]   # (T,L,F)
        y_link = self.link_labels[idx]
        y_motor = self.motor_labels[idx]
        return (
            torch.from_numpy(node_w),
            torch.tensor(y_link, dtype=torch.int64),
            torch.tensor(y_motor, dtype=torch.int64),
        )

def compute_norm_stats(node_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = node_features.reshape(-1, node_features.shape[-1])
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def normalize_inplace(node_features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    node_features -= mean
    node_features /= std

# ============================ Positional encoding ============================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard transformer sinusoidal PE."""
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, t_len: int) -> torch.Tensor:
        """returns (1, T, 1, D) slice for broadcasting"""
        return self.pe[:t_len].unsqueeze(0).unsqueeze(2)  # (1,T,1,D)

# ============================ Transformer Model ============================

class TemporalLinkTransformer(nn.Module):
    """
    Tokens = (time, link) pairs, plus [CLS].
    Output: link logits (B,L+1), motor logits (B,8L+1)
    OOM-safe: gradient checkpointing, SDPA kernel (new API) with fallback-to-noop
    """
    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        link_count: int,
        use_checkpoint: bool = False,
        use_sdp: bool = True,
    ) -> None:
        super().__init__()
        self.L = link_count
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.use_sdp = use_sdp

        self.in_proj = nn.Linear(feature_dim, d_model)
        self.link_embed = nn.Embedding(link_count, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
            ) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Learned [CLS]
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Heads
        self.link_node_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.link_healthy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.motor_cond_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, MOTORS_PER_LINK),
        )
        self.motor_healthy_bias = nn.Parameter(torch.tensor(0.0))

    def _sdp_ctx(self):
        # Prefer new API (PyTorch >= 2.5). If unavailable, do nothing (avoid deprecated warnings).
        if not (self.use_sdp and torch.cuda.is_available()):
            return nullcontext()
        try:
            import torch.nn.attention as attn
            if hasattr(attn, "sdpa_kernel") and hasattr(attn, "SDPBackend"):
                FLASH = getattr(attn.SDPBackend, "FLASH_ATTENTION", None)
                MEM   = getattr(attn.SDPBackend, "MEMORY_EFFICIENT", None)
                if FLASH is not None and MEM is not None:
                    return attn.sdpa_kernel(FLASH, MEM)
        except Exception:
            pass
        return nullcontext()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        import torch.utils.checkpoint as cp
        with self._sdp_ctx():
            h = x
            for layer in self.layers:
                if self.use_checkpoint and self.training:
                    h = cp.checkpoint(layer, h, use_reentrant=False)
                else:
                    h = layer(h)
            h = self.final_norm(h)
        return h

    def forward(self, node_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        node_seq: (B, T, L, F)
        returns: link_logits (B, L+1), motor_logits (B, 8L+1)
        """
        B, T, L, F = node_seq.shape
        assert L == self.L, f"Link count mismatch: {L} != {self.L}"

        x = self.in_proj(node_seq)          # (B,T,L,D)
        link_ids = torch.arange(L, device=x.device, dtype=torch.long).view(1, 1, L)
        x = x + self.link_embed(link_ids).to(x.dtype)  # (B,T,L,D)
        pe = self.posenc(T).to(x.dtype).to(x.device)   # (1,T,1,D)
        x = x + pe

        x = x.reshape(B, T * L, self.d_model)          # (B, TL, D)
        cls = self.cls.expand(B, 1, self.d_model)      # (B,1,D)
        x = torch.cat([cls, x], dim=1)                 # (B, 1+TL, D)

        h = self._encode(x)                            # (B, 1+TL, D)
        h_cls = h[:, 0]                                # (B,D)
        start = 1 + (T - 1) * L
        h_last = h[:, start : start + L]               # (B,L,D)

        node_scores = self.link_node_head(h_last).squeeze(-1)        # (B,L)
        healthy = self.link_healthy_head(h_cls).squeeze(-1)          # (B,)
        link_logits = torch.cat([healthy.unsqueeze(-1), node_scores], dim=-1)  # (B,L+1)

        cond = self.motor_cond_head(h_last)                          # (B,L,8)
        combined = cond + link_logits[:, 1:].unsqueeze(-1)           # (B,L,8)
        flat = combined.reshape(B, L * MOTORS_PER_LINK)              # (B,8L)
        m0 = link_logits[:, 0:1] + self.motor_healthy_bias           # (B,1)
        motor_logits = torch.cat([m0, flat], dim=1)                  # (B,8L+1)
        return link_logits, motor_logits

# ============================ Losses/metrics ============================

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def hard_mask_motor_logits(motor_logits: torch.Tensor, link_idx: torch.Tensor, L: int) -> torch.Tensor:
    """
    Eval-time structural mask: keep only motors from predicted link (and healthy=0).
    (FP16/AMP 안전) dtype별 최소 유한값으로 마스킹하여 오버플로우 방지.
    """
    B = motor_logits.size(0)
    device = motor_logits.device
    idx = torch.arange(0, L * MOTORS_PER_LINK + 1, device=device)
    glink = torch.zeros_like(idx)
    if L > 0:
        glink[1:] = ((idx[1:] - 1) // MOTORS_PER_LINK) + 1

    allowed = (glink.unsqueeze(0) == link_idx.unsqueeze(1)) | (idx.unsqueeze(0) == 0)
    allowed = allowed.to(torch.bool)

    # dtype-safe large negative for masking (e.g., -65504 for float16)
    neg_large = torch.tensor(torch.finfo(motor_logits.dtype).min, dtype=motor_logits.dtype, device=device)
    return torch.where(allowed, motor_logits, neg_large)

def consistency_loss(link_logits: torch.Tensor, motor_logits: torch.Tensor, L: int) -> torch.Tensor:
    """
    KL(link_from_motor || link_head) + healthy prob L1.
    FP16/FP32 모두 안전.
    """
    if L == 0:
        return motor_logits.new_zeros(())
    B = motor_logits.size(0)
    # motors -> links (log-sum-exp over 8 motors per link)
    link_from_motor = motor_logits.new_empty(B, L)
    for i in range(L):
        s = 1 + i * MOTORS_PER_LINK
        e = s + MOTORS_PER_LINK
        link_from_motor[:, i] = torch.logsumexp(motor_logits[:, s:e], dim=1)

    # KL(q||p) with q = Softmax(link_from_motor), p = Softmax(link_logits[:,1:])
    log_q = F.log_softmax(link_from_motor, dim=1)
    p     = F.softmax(link_logits[:, 1:], dim=1)
    kl = F.kl_div(log_q, p, reduction="batchmean")

    # healthy prob alignment (class 0)
    p0 = F.softmax(link_logits, dim=1)[:, 0]
    m0 = F.softmax(motor_logits, dim=1)[:, 0]
    healthy_l1 = F.l1_loss(m0, p0)
    return kl + healthy_l1

# ---- metrics helpers (macro/micro; with/without healthy=0) ----

def _init_counts(C: int) -> Dict[str, np.ndarray]:
    return {"tp": np.zeros(C, dtype=np.int64),
            "fp": np.zeros(C, dtype=np.int64),
            "fn": np.zeros(C, dtype=np.int64)}

def _update_counts(counts: Dict[str, np.ndarray], y_pred: torch.Tensor, y_true: torch.Tensor, C: int) -> None:
    yp = y_pred.detach().to("cpu").numpy()
    yt = y_true.detach().to("cpu").numpy()
    for c in range(C):
        pred_c = (yp == c)
        true_c = (yt == c)
        tp = np.logical_and(pred_c, true_c).sum()
        fp = np.logical_and(pred_c, ~true_c).sum()
        fn = np.logical_and(~pred_c, true_c).sum()
        counts["tp"][c] += int(tp)
        counts["fp"][c] += int(fp)
        counts["fn"][c] += int(fn)

def _compute_prf(counts: Dict[str, np.ndarray], exclude_zero: bool = False) -> Dict[str, float]:
    tp, fp, fn = counts["tp"].astype(np.float64), counts["fp"].astype(np.float64), counts["fn"].astype(np.float64)
    C = tp.shape[0]
    class_mask = (tp + fn) > 0  # only classes with support
    if exclude_zero and C > 0:
        class_mask[0] = False

    with np.errstate(divide='ignore', invalid='ignore'):
        prec_c = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rec_c  = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1_c   = np.divide(2 * prec_c * rec_c, (prec_c + rec_c),
                           out=np.zeros_like(prec_c), where=(prec_c + rec_c) > 0)

    sel = class_mask
    sel_count = max(int(sel.sum()), 1)
    macro_p = float(prec_c[sel].sum() / sel_count)
    macro_r = float(rec_c[sel].sum() / sel_count)
    macro_f1 = float(f1_c[sel].sum() / sel_count)

    TP = tp.sum(); FP = fp.sum(); FN = fn.sum()
    micro_p = float(TP / max(TP + FP, 1e-9))
    micro_r = float(TP / max(TP + FN, 1e-9))
    micro_f1 = float(0.0 if (micro_p + micro_r) == 0 else (2 * micro_p * micro_r) / (micro_p + micro_r))

    return {
        "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }

# ============================ Train loop ============================

def run_epoch(
    model: TemporalLinkTransformer,
    loader: DataLoader,
    link_loss_fn: nn.Module,
    motor_loss_fn: nn.Module,
    motor_weight: float,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    amp: bool,
    link_count: int,
    hard_mask_eval: bool,
    consistency_weight: float,
    accum_steps: int = 1,
    compute_metrics: bool = True,
) -> Tuple[float, float, float, Dict[str, float]]:
    training = optimizer is not None

    # autocast for CUDA when amp=True
    autocast_cm = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if (amp and device.type == "cuda") else nullcontext()
    # inference_mode for eval reduces mem/overhead
    outer_cm = torch.inference_mode if not training else nullcontext

    if training:
        model.train(True)
    else:
        model.train(False)

    total_loss = 0.0
    total_link_correct = 0
    total_motor_correct = 0
    total_samples = 0

    if compute_metrics:
        C_link = link_count + 1
        C_motor = link_count * MOTORS_PER_LINK + 1
        link_counts = _init_counts(C_link)
        motor_counts = _init_counts(C_motor)
    else:
        link_counts = motor_counts = None

    if training:
        optimizer.zero_grad(set_to_none=True)

    with outer_cm():
        for step, (node_win, y_link, y_motor) in enumerate(loader):
            node_win = node_win.to(device, non_blocking=True).float()
            y_link = y_link.to(device, non_blocking=True)
            y_motor = y_motor.to(device, non_blocking=True)

            with autocast_cm:
                link_logits, motor_logits = model(node_win)
                loss_link  = link_loss_fn(link_logits, y_link)
                loss_motor = motor_loss_fn(motor_logits, y_motor)
                loss = loss_link + motor_weight * loss_motor
                if consistency_weight > 0.0:
                    loss = loss + consistency_weight * consistency_loss(link_logits, motor_logits, link_count)

            bsz = y_link.size(0)

            if training:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * bsz
            link_pred = link_logits.argmax(dim=1)
            if not training and hard_mask_eval:
                masked = hard_mask_motor_logits(motor_logits, link_pred, link_count)
                motor_pred = masked.argmax(dim=1)
            else:
                motor_pred = motor_logits.argmax(dim=1)
            total_link_correct  += (link_pred  == y_link).sum().item()
            total_motor_correct += (motor_pred == y_motor).sum().item()
            total_samples += bsz

            if compute_metrics:
                _update_counts(link_counts, link_pred, y_link, C_link)
                _update_counts(motor_counts, motor_pred, y_motor, C_motor)

    mean_loss = total_loss / max(total_samples, 1)
    link_acc  = total_link_correct / max(total_samples, 1)
    motor_acc = total_motor_correct / max(total_samples, 1)

    metrics = {}
    if compute_metrics:
        link_all = _compute_prf(link_counts, exclude_zero=False)
        link_fault = _compute_prf(link_counts, exclude_zero=True)
        motor_all = _compute_prf(motor_counts, exclude_zero=False)
        motor_fault = _compute_prf(motor_counts, exclude_zero=True)
        metrics = {
            "link_macro_p": link_all["macro_p"], "link_macro_r": link_all["macro_r"], "link_macro_f1": link_all["macro_f1"],
            "link_fault_macro_p": link_fault["macro_p"], "link_fault_macro_r": link_fault["macro_r"], "link_fault_macro_f1": link_fault["macro_f1"],
            "link_micro_f1": link_all["micro_f1"],
            "motor_macro_p": motor_all["macro_p"], "motor_macro_r": motor_all["macro_r"], "motor_macro_f1": motor_all["macro_f1"],
            "motor_fault_macro_p": motor_fault["macro_p"], "motor_fault_macro_r": motor_fault["macro_r"], "motor_fault_macro_f1": motor_fault["macro_f1"],
            "motor_micro_f1": motor_all["micro_f1"],
        }

    return mean_loss, link_acc, motor_acc, metrics

# ============================ CLI & main ============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer FDI Trainer (accuracy-first, OOM-safe)")
    p.add_argument("--data-root", type=str, default="data_storage/link_3")
    p.add_argument("--max-shards", type=int, default=0)
    p.add_argument("--baseline-window", type=int, default=120)
    p.add_argument("--include-motor-features", action="store_true", default=True)
    p.add_argument("--component-threshold-deg", type=float, default=18.0)

    p.add_argument("--window", type=int, default=96)
    p.add_argument("--pos-stride", type=int, default=3)
    p.add_argument("--neg-stride", type=int, default=12)
    p.add_argument("--val-ratio", type=float, default=0.2)

    # Transformer size
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--ffn-size", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # Train
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument("--motor-weight", type=float, default=1.0)
    p.add_argument("--label-smoothing-link", type=float, default=0.05)
    p.add_argument("--label-smoothing-motor", type=float, default=0.05)

    # Consistency & eval mask
    p.add_argument("--consistency-weight", type=float, default=0.2)
    p.add_argument("--hard-mask-eval", action="store_true")

    # OOM-safe toggles
    p.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing per Transformer layer.")
    p.add_argument("--no-sdp", action="store_true", help="Disable SDPA Flash/Mem-Efficient attention kernels.")

    # Metrics control
    p.add_argument("--metrics-train", action="store_true", default=True, help="Compute and print train PR/REC/F1 each epoch.")

    p.add_argument("--seed", type=int, default=2024)

    # Save paths
    p.add_argument("--save", type=str, default=None, help="Base path. If set, saves best_*.pt and last_*.pt in the same folder.")
    p.add_argument("--save-best", type=str, default=None, help="Explicit path for best checkpoint.")
    p.add_argument("--save-last", type=str, default=None, help="Explicit path for last checkpoint.")

    # Engine
    p.add_argument("--amp", action="store_true", default=True, help="Use AMP in training.")
    p.add_argument("--eval-amp", action="store_true", default=True, help="Use AMP in evaluation.")
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--persistent-workers", action="store_true", default=True)
    return p.parse_args()

def _derive_save_paths(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    # Priority: explicit paths
    if args.save_best or args.save_last:
        return args.save_best, args.save_last
    if not args.save:
        return None, None
    p = Path(args.save)
    parent, stem, suf = p.parent, p.stem, (p.suffix or ".pt")
    best_path = str(parent / f"best_{stem}{suf}")
    last_path = str(parent / f"last_{stem}{suf}")
    return best_path, last_path

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")

    # TF32 (Ampere+)
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    shard_paths = discover_shards(args.data_root)
    if not shard_paths:
        raise FileNotFoundError(f"No shards found at {args.data_root}")
    if args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]
    print(f"[data] found {len(shard_paths)} shards")

    pack = load_all_shards(
        shard_paths,
        baseline_window=args.baseline_window,
        include_motor_features=args.include_motor_features,
        component_threshold_deg=args.component_threshold_deg,
    )
    node_all = pack.node_features
    y_link_all  = pack.link_labels
    y_motor_all = pack.motor_labels
    L = pack.link_count
    F_node = node_all.shape[-1]
    print(f"[data] nodes {node_all.shape}, link_labels {y_link_all.shape}, motor_labels {y_motor_all.shape} | links={L}")

    # Split by sequence
    rng = np.random.default_rng(args.seed)
    N = node_all.shape[0]
    perm = rng.permutation(N)
    val_count = max(1, int(N * args.val_ratio))
    val_idx = perm[:val_count]
    tr_idx  = perm[val_count:]

    node_tr, node_val = node_all[tr_idx], node_all[val_idx]
    y_link_tr, y_link_val = y_link_all[tr_idx], y_link_all[val_idx]
    y_motor_tr, y_motor_val = y_motor_all[tr_idx], y_motor_all[val_idx]

    # Normalize
    mean, std = compute_norm_stats(node_tr)
    normalize_inplace(node_tr, mean, std)
    normalize_inplace(node_val, mean, std)

    # Datasets/loaders
    train_ds = FeatureWindowDataset(
        node_tr, y_link_tr, y_motor_tr,
        window=args.window, pos_stride=args.pos_stride, neg_stride=args.neg_stride
    )
    val_ds = FeatureWindowDataset(
        node_val, y_link_val, y_motor_val,
        window=args.window, pos_stride=args.pos_stride, neg_stride=args.neg_stride
    )
    print(f"[data] windows train={len(train_ds)} val={len(val_ds)}")

    pin = device.type == "cuda"
    nw = args.num_workers
    pf = args.prefetch_factor if nw and args.prefetch_factor > 0 else None
    pw = args.persistent_workers and (nw > 0)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=nw, pin_memory=pin, persistent_workers=pw, prefetch_factor=pf
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False, drop_last=False,
        num_workers=nw, pin_memory=pin, persistent_workers=pw, prefetch_factor=pf
    )

    # Model
    model = TemporalLinkTransformer(
        feature_dim=F_node,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ffn_size,
        dropout=args.dropout,
        link_count=L,
        use_checkpoint=args.grad_checkpoint,
        use_sdp=(not args.no_sdp),
    ).to(device)

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[warn] torch.compile failed ({e}); fallback to eager.", flush=True)

    # Losses & optimizer
    num_link_classes  = L + 1
    num_motor_classes = L * MOTORS_PER_LINK + 1
    link_w  = compute_class_weights(train_ds.link_labels,  num_link_classes).to(device)
    motor_w = compute_class_weights(train_ds.motor_labels, num_motor_classes).to(device)

    link_loss_fn = nn.CrossEntropyLoss(weight=link_w, label_smoothing=args.label_smoothing_link)
    motor_loss_fn = nn.CrossEntropyLoss(weight=motor_w, label_smoothing=args.label_smoothing_motor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric = -1.0
    best_epoch = 0
    best_path, last_path = _derive_save_paths(args)

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_link_acc, tr_motor_acc, tr_m = run_epoch(
            model, train_loader, link_loss_fn, motor_loss_fn,
            args.motor_weight, optimizer, device, amp=args.amp, link_count=L,
            hard_mask_eval=False, consistency_weight=args.consistency_weight,
            accum_steps=args.accum_steps, compute_metrics=args.metrics_train,
        )
        val_loss, val_link_acc, val_motor_acc, va_m = run_epoch(
            model, val_loader, link_loss_fn, motor_loss_fn,
            args.motor_weight, optimizer=None, device=device, amp=args.eval_amp, link_count=L,
            hard_mask_eval=args.hard_mask_eval, consistency_weight=args.consistency_weight,
            accum_steps=1, compute_metrics=True,
        )
        scheduler.step()

        # Choose best by fault-only link macro F1
        current_score = va_m["link_fault_macro_f1"]

        if current_score > best_metric:
            best_metric = current_score
            best_epoch = ep
            if best_path:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "node_mean": mean,
                        "node_std":  std,
                        "config": vars(args),
                        "val_link_acc": float(val_link_acc),
                        "val_link_fault_macro_f1": float(current_score),
                        "link_count": L,
                        "feature_dim": F_node,
                        "d_model": args.d_model,
                        "epoch": ep,
                        "metrics": va_m,
                    },
                    best_path,
                )

        def fmt(m, k): return f"{m[k]*100:5.2f}%"
        tr_line = (f"train: loss={tr_loss:.4f} | acc(link/motor)={tr_link_acc*100:5.2f}%/{tr_motor_acc*100:5.2f}%")
        if args.metrics_train:
            tr_line += (f" | L-mF1(all/fault)={fmt(tr_m,'link_macro_f1')}/{fmt(tr_m,'link_fault_macro_f1')}"
                        f" | M-mF1(all/fault)={fmt(tr_m,'motor_macro_f1')}/{fmt(tr_m,'motor_fault_macro_f1')}")
        va_line = (f"val  : loss={val_loss:.4f} | acc(link/motor)={val_link_acc*100:5.2f}%/{val_motor_acc*100:5.2f}%"
                   f" | L-mP/R/F1={fmt(va_m,'link_macro_p')}/{fmt(va_m,'link_macro_r')}/{fmt(va_m,'link_macro_f1')}"
                   f" | L-fault-mF1={fmt(va_m,'link_fault_macro_f1')}"
                   f" | M-mF1={fmt(va_m,'motor_macro_f1')} | M-fault-mF1={fmt(va_m,'motor_fault_macro_f1')}"
                   f" | best L-fault-mF1={best_metric*100:5.2f}% (ep{best_epoch})")
        print(f"Epoch {ep:02d}/{args.epochs} | {tr_line}\n{va_line}")

    # Save last
    if last_path:
        torch.save(
            {
                "model_state": model.state_dict(),
                "node_mean": mean,
                "node_std":  std,
                "config": vars(args),
                "link_count": L,
                "feature_dim": F_node,
                "d_model": args.d_model,
                "best_epoch": best_epoch,
                "best_link_fault_macro_f1": float(best_metric),
            },
            last_path,
        )

    print(f"[done] best fault-only link macro-F1 {best_metric*100:.2f}% at epoch {best_epoch}")
    if best_path:
        print(f"[save] best -> {best_path}")
    if last_path:
        print(f"[save] last -> {last_path}")

if __name__ == "__main__":
    main()


"""
python3 transformer_fdi_train.py \
  --data-root data_storage/link_3 \
  --epochs 50 \
  --window 128 --pos-stride 2 --neg-stride 8 \
  --d-model 512 --nhead 8 --layers 8 --ffn-size 2048 \
  --batch-size 96 --val-batch-size 96 \
  --include-motor-features \
  --consistency-weight 0.3 --hard-mask-eval \
  --num-workers 12 --prefetch-factor 6 --persistent-workers \
  --amp --eval-amp \
  --grad-checkpoint \
  --save trans_fdi_big.pt
"""
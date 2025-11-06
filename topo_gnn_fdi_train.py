#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a real-time LASDRA fault detector using topological SE(3) graph features
(GNN version with hierarchical link→motor constraint, accelerated).

- Keeps the original data I/O contract (npz shards).
- Node/edge features follow the topological design (SO(3)/S^2/Lie + continuity).
- Backbone: causal GraphGRU with edge gating.
- Two heads with STRUCTURAL COUPLING:
    (A) Link classifier (0=healthy, 1..L = faulty link id)
    (B) Motor classifier (0=healthy, 1..8L = global motor id)
    Motor logits are built HIERARCHICALLY from link logits + per-link (8) motor logits.
- Optional CONSISTENCY LOSS between link head and motor-induced link distribution.
- Optional HARD-MASK inference for motors using predicted link (post-process).

Speed:
- AMP (torch.amp), vectorized feature builder, worker prefetch, optional torch.compile.

Author: (you)
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================ Constants & small utils ============================

MOTORS_PER_LINK = 8
EPS = 1e-9

# ------------------------- vector/matrix helpers -------------------------

def _unitize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, EPS)
    return vec / norm

def _so3_log_batch(R: np.ndarray) -> np.ndarray:
    """
    Batched SO(3) logarithm → vee(skew) * scale.
    R: (..., 3, 3)
    returns w: (..., 3)
    """
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

def _so3_axis_from_R(R: np.ndarray) -> np.ndarray:
    """Axis on S^2 from rotation matrix via log(R)."""
    w = _so3_log_batch(R)
    axis = _unitize(w)
    return axis

# ------------------------- SE(3) residuals (desired vs actual) -------------------------

def compute_se3_residual(desired: np.ndarray, actual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    desired, actual: (..., 4,4)
    return:
      omega_resid: (..., 3)  rotation residual (so3 log of R_d^T R_a)
      pos_err:     (..., 3)  translation residual p_a - p_d
    """
    pos_des = desired[..., :3, 3]
    pos_act = actual[..., :3, 3]
    pos_err = pos_act - pos_des

    rot_des = desired[..., :3, :3]
    rot_act = actual[..., :3, :3]
    rot_err = np.matmul(np.swapaxes(rot_des, -1, -2), rot_act)
    omega = _so3_log_batch(rot_err)
    return omega.astype(np.float32, copy=False), pos_err.astype(np.float32, copy=False)

# ------------------------- Motor geometry on the chain -------------------------

def default_motor_layout(link_count: int) -> np.ndarray:
    """
    Simple canonical layout of 8 motors per link (unitized directions in link frame).
    Returns: (L, 8, 3)
    """
    base = np.array(
        [
            [1.0, 0.0, 0.5],
            [-1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, -1.0, 0.5],
            [0.5, 0.5, 1.0],
            [-0.5, 0.5, 1.0],
            [0.5, -0.5, 1.0],
            [-0.5, -0.5, 1.0],
        ],
        dtype=np.float32,
    )
    layout = np.tile(_unitize(base)[None, :, :], (link_count, 1, 1))
    return layout

def compute_motor_directions(rotations: np.ndarray, motor_layout: np.ndarray) -> np.ndarray:
    """
    rotations: (S,T,L,3,3)
    motor_layout: (L,8,3)
    returns: (S,T,L,8,3) in world frame
    """
    dirs = np.einsum("stlij,lmj->stlmi", rotations, motor_layout, optimize=True)
    return _unitize(dirs.astype(np.float32, copy=False))

def count_components(directions: np.ndarray, cos_thresh: float) -> int:
    """Connected components on thresholded cosine graph over 8 motor directions (BFS)."""
    M = directions.shape[0]
    visited = np.zeros(M, dtype=bool)
    comps = 0
    for i in range(M):
        if visited[i]:
            continue
        comps += 1
        queue = [i]
        visited[i] = True
        while queue:
            idx = queue.pop()
            dots = np.sum(directions[idx] * directions, axis=-1)
            neighbors = np.where((dots >= cos_thresh) & (~visited))[0]
            if neighbors.size == 0:
                continue
            visited[neighbors] = True
            queue.extend(neighbors.tolist())
    return comps

# ------------------------- Labels collapsing (same as provided) -------------------------

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

# ============================ Graph feature pack ============================

@dataclass
class GraphFeaturePack:
    node_features: np.ndarray  # (S,T,L,F_node)
    edge_features: np.ndarray  # (S,T,L-1,F_edge)
    link_labels:   np.ndarray  # (S,T)
    motor_labels:  np.ndarray  # (S,T)
    node_dim: int
    edge_dim: int
    link_count: int

# ============================ Feature builder (vectorized) ============================

def build_graph_features(
    desired_cum: np.ndarray,  # (S,T,L,4,4)
    actual_cum:  np.ndarray,  # (S,T,L,4,4)
    labels:      np.ndarray,  # (S,T,8L) 1=healthy,0=fault
    baseline_window: int = 120,
    component_threshold_deg: float = 18.0,
    per_motor_mode: str = "full",           # "full" or "none"
    component_metric: str = "density",      # "density" (fast) or "count" (slow, BFS)
) -> GraphFeaturePack:
    S, T, L, _, _ = desired_cum.shape

    # Residuals & energy
    omega_resid, pos_err = compute_se3_residual(desired_cum, actual_cum)  # (S,T,L,3)
    energy = (omega_resid**2 + pos_err**2).sum(axis=-1, keepdims=True).astype(np.float32)

    # Absolute rotations, axes
    R_act = actual_cum[..., :3, :3]                          # (S,T,L,3,3)
    axis_i = _so3_axis_from_R(R_act).astype(np.float32)      # (S,T,L,3)

    # Temporal angular velocity
    omega_abs = np.zeros((S, T, L, 3), dtype=np.float32)
    if T > 1:
        R_t  = R_act[:, :-1]
        R_t1 = R_act[:, 1:]
        Rt = np.matmul(np.swapaxes(R_t, -1, -2), R_t1)       # (S,T-1,L,3,3)
        omega_abs[:, :-1] = _so3_log_batch(Rt).astype(np.float32, copy=False)

    # Edge features: xi, alignment a
    xi = np.zeros((S, T, L-1, 3), dtype=np.float32)
    a  = np.zeros((S, T, L-1, 1), dtype=np.float32)
    if L > 1:
        R_i   = R_act[:, :, :-1]
        R_ip1 = R_act[:, :, 1:]
        Rij   = np.matmul(np.swapaxes(R_i, -1, -2), R_ip1)
        xi    = _so3_log_batch(Rij).astype(np.float32, copy=False)
        ri = axis_i[:, :, :-1, :]
        rj = axis_i[:, :, 1:,  :]
        a  = (1.0 - np.sum(ri * rj, axis=-1, keepdims=True)).astype(np.float32)

    # Continuity c
    c = np.zeros((S, T, max(L-1,0), 1), dtype=np.float32)
    if T > 1 and L > 1:
        diff = xi[:, 1:] - xi[:, :-1]
        c[:, 1:] = np.linalg.norm(diff, axis=-1, keepdims=True)

    # Motor topology summaries
    motor_layout = default_motor_layout(L)                         # (L,8,3)
    motor_dirs   = compute_motor_directions(R_act, motor_layout)   # (S,T,L,8,3)
    cos_thr = math.cos(math.radians(component_threshold_deg))

    if per_motor_mode == "none":
        component_norm = np.zeros((S, T, L, 1), dtype=np.float32)
        per_motor_flat = np.zeros((S, T, L, 0), dtype=np.float32)
    else:
        base_window = max(1, min(baseline_window, T))
        baseline_dirs = _unitize(motor_dirs[:, :base_window].mean(axis=1, keepdims=False))  # (S,L,8,3)

        m_now  = motor_dirs                                 # (S,T,L,8,3)
        m_base = baseline_dirs[:, None]                     # (S,1,L,8,3)
        dots = np.clip((m_now * m_base).sum(axis=-1), -1.0, 1.0)      # (S,T,L,8)
        ang  = np.arccos(dots)[..., None]                              # (S,T,L,8,1)
        diff = (m_now - m_base)                                        # (S,T,L,8,3)
        per_motor_flat = np.concatenate([m_now, diff, ang], axis=-1)   # (S,T,L,8,7)
        per_motor_flat = per_motor_flat.reshape(S, T, L, MOTORS_PER_LINK * 7).astype(np.float32, copy=False)

        if component_metric == "density":
            cos = np.einsum("stlmk,stlpk->stlmp", m_now, m_now, optimize=True)  # (S,T,L,8,8)
            upper = np.triu(np.ones((MOTORS_PER_LINK, MOTORS_PER_LINK), dtype=bool), k=1)
            edges = (cos >= cos_thr)[..., upper].sum(axis=-1)  # (S,T,L)
            denom = MOTORS_PER_LINK * (MOTORS_PER_LINK - 1) / 2.0
            density = (edges / denom).astype(np.float32, copy=False)
            component_norm = density[..., None]
        else:
            component_norm = np.empty((S, T, L, 1), dtype=np.float32)
            for s in range(S):
                for t in range(T):
                    for l in range(L):
                        comp = count_components(m_now[s, t, l], cos_thr)
                        component_norm[s, t, l, 0] = comp / float(MOTORS_PER_LINK)

    # Pack node/edge features
    node_parts = [
        omega_resid.astype(np.float32, copy=False),
        pos_err.astype(np.float32, copy=False),
        energy,
        axis_i,
        omega_abs,
        component_norm,
    ]
    if per_motor_flat.shape[-1] > 0:
        node_parts.append(per_motor_flat)

    node_features = np.concatenate(node_parts, axis=-1)
    if L > 1:
        edge_features = np.concatenate([xi.astype(np.float32, copy=False), c, a], axis=-1)
    else:
        edge_features = np.zeros((S, T, 0, 5), dtype=np.float32)

    link_labels  = build_link_targets(labels)
    motor_labels = build_motor_targets(labels)

    return GraphFeaturePack(
        node_features=node_features,
        edge_features=edge_features,
        link_labels=link_labels,
        motor_labels=motor_labels,
        node_dim=node_features.shape[-1],
        edge_dim=edge_features.shape[-1] if L > 1 else 5,
        link_count=L,
    )

# ============================ Shard discovery & loader ============================

def discover_shards(root: str) -> List[Path]:
    pattern = str(Path(root) / "fault_dataset_shard_*.npz")
    return [Path(p) for p in sorted(glob.glob(pattern))]

def load_all_shards(
    paths: Sequence[Path],
    baseline_window: int,
    component_threshold_deg: float,
    per_motor_mode: str = "full",
    component_metric: str = "density",
) -> GraphFeaturePack:
    node_buf: List[np.ndarray] = []
    edge_buf: List[np.ndarray] = []
    link_buf: List[np.ndarray] = []
    motor_buf: List[np.ndarray] = []
    node_dim = None
    edge_dim = None
    link_count = None

    total = len(paths)
    for idx, path in enumerate(paths, 1):
        with np.load(path) as data:
            desired_cum = data["desired_link_cum"]
            actual_cum  = data["actual_link_cum"]
            labels      = data["label"]

        pack = build_graph_features(
            desired_cum=desired_cum,
            actual_cum=actual_cum,
            labels=labels,
            baseline_window=baseline_window,
            component_threshold_deg=component_threshold_deg,
            per_motor_mode=per_motor_mode,
            component_metric=component_metric,
        )
        node_buf.append(pack.node_features)
        edge_buf.append(pack.edge_features)
        link_buf.append(pack.link_labels)
        motor_buf.append(pack.motor_labels)

        if node_dim is None:
            node_dim = pack.node_dim
            edge_dim = pack.edge_dim
            link_count = pack.link_count
        else:
            if node_dim != pack.node_dim or edge_dim != pack.edge_dim:
                raise ValueError("Inconsistent feature dimensions across shards.")
            if link_count != pack.link_count:
                raise ValueError("Inconsistent link count across shards.")

        print(f"[data] processed shard {idx}/{total}", flush=True)

    node_features = np.concatenate(node_buf, axis=0)
    edge_features = np.concatenate(edge_buf, axis=0)
    link_labels   = np.concatenate(link_buf,  axis=0)
    motor_labels  = np.concatenate(motor_buf, axis=0)

    return GraphFeaturePack(
        node_features=node_features,
        edge_features=edge_features,
        link_labels=link_labels,
        motor_labels=motor_labels,
        node_dim=node_dim or 0,
        edge_dim=edge_dim or 0,
        link_count=link_count or 0,
    )

# ============================ Dataset & normalization ============================

class GraphWindowDataset(Dataset):
    """
    Returns causal windows:
      - node_window: (T, L, F_node)
      - edge_window: (T, L-1, F_edge)
      - link_label:  int (last time step)
      - motor_label: int (last time step)
    """
    def __init__(
        self,
        node_features: np.ndarray,  # (N,T,L,F_node)
        edge_features: np.ndarray,  # (N,T,L-1,F_edge)
        link_labels:   np.ndarray,  # (N,T)
        motor_labels:  np.ndarray,  # (N,T)
        window: int,
        pos_stride: int,
        neg_stride: int,
    ) -> None:
        if node_features.shape[:2] != link_labels.shape[:2]:
            raise ValueError("node_features and link_labels shape mismatch.")
        if edge_features.shape[0] != node_features.shape[0] or edge_features.shape[1] != node_features.shape[1]:
            raise ValueError("edge_features and node_features must share (N,T).")
        if window <= 0 or window > node_features.shape[1]:
            raise ValueError("invalid window length.")

        self.node = node_features
        self.edge = edge_features
        self.link_full = link_labels
        self.motor_full = motor_labels
        self.window = window

        N, T, L, _ = node_features.shape
        slots_seq: List[int] = []
        slots_start: List[int] = []
        slots_link: List[int] = []
        slots_motor: List[int] = []

        for seq in range(N):
            end = window
            while end <= T:
                start = end - window
                link_y = int(link_labels[seq, end - 1])
                motor_y = int(motor_labels[seq, end - 1])
                slots_seq.append(seq)
                slots_start.append(start)
                slots_link.append(link_y)
                slots_motor.append(motor_y)
                step = pos_stride if link_y != 0 else neg_stride
                end += step

        self.seq_idx = np.asarray(slots_seq, dtype=np.int32)
        self.start_idx = np.asarray(slots_start, dtype=np.int32)
        self.link_labels = np.asarray(slots_link, dtype=np.int64)
        self.motor_labels = np.asarray(slots_motor, dtype=np.int64)

    def __len__(self) -> int:
        return self.link_labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.seq_idx[idx]
        start = self.start_idx[idx]
        end = start + self.window
        node_w = self.node[seq, start:end]   # (T,L,F_node)
        edge_w = self.edge[seq, start:end]   # (T,L-1,F_edge)
        link_label = self.link_labels[idx]
        motor_label = self.motor_labels[idx]
        return (
            torch.from_numpy(node_w),
            torch.from_numpy(edge_w),
            torch.tensor(link_label, dtype=torch.int64),
            torch.tensor(motor_label, dtype=torch.int64),
        )

def compute_norm_stats_nodes_edges(
    node_features: np.ndarray, edge_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-dimension mean/std for node and edge features separately."""
    flat_node = node_features.reshape(-1, node_features.shape[-1])
    flat_edge = edge_features.reshape(-1, edge_features.shape[-1])
    n_mean = flat_node.mean(axis=0)
    n_std  = flat_node.std(axis=0)
    e_mean = flat_edge.mean(axis=0)
    e_std  = flat_edge.std(axis=0)
    n_std[n_std < 1e-6] = 1e-6
    e_std[e_std < 1e-6] = 1e-6
    return n_mean.astype(np.float32), n_std.astype(np.float32), e_mean.astype(np.float32), e_std.astype(np.float32)

def normalize_inplace_nodes_edges(
    node_features: np.ndarray, edge_features: np.ndarray,
    n_mean: np.ndarray, n_std: np.ndarray, e_mean: np.ndarray, e_std: np.ndarray
) -> None:
    node_features -= n_mean
    node_features /= n_std
    edge_features -= e_mean
    edge_features /= e_std

# ============================ GraphGRU model (with constraints) ============================

class GraphGRUCell(nn.Module):
    """
    One graph-recurrent step with dynamic edge gating:
      - Edge MLP maps edge feature to weight in (0,1): w_e(t)
      - Aggregation per node i: m_i = (w_self*h_i + w_left*h_{i-1} + w_right*h_{i+1}) / deg_i
      - GRUCell update on x_i (node feat) + message m_i
    """
    def __init__(self, node_in: int, edge_in: int, hidden: int, self_loop_weight: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.self_w = float(self_loop_weight)
        self.node_enc = nn.Linear(node_in, hidden)
        self.msg_enc  = nn.Linear(hidden, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_prev: torch.Tensor, node_x: torch.Tensor, edge_x: torch.Tensor) -> torch.Tensor:
        """
        h_prev: (B, L, H)
        node_x: (B, L, F_node)
        edge_x: (B, L-1, F_edge)  for edges (i,i+1)
        """
        B, L, H = h_prev.shape

        if L > 1:
            w_e = self.edge_mlp(edge_x).squeeze(-1)  # (B, L-1)
            w_left  = F.pad(w_e, (1, 0), value=0.0)  # (B, L): edge (i-1)->i
            w_right = F.pad(w_e, (0, 1), value=0.0)  # (B, L): edge i->(i+1)
            h_left  = F.pad(h_prev, (0, 0, 1, 0))[:, :-1, :]  # (B,L,H)
            h_right = F.pad(h_prev, (0, 0, 0, 1))[:, 1:, :]   # (B,L,H)
        else:
            w_left  = torch.zeros(B, L, device=h_prev.device, dtype=h_prev.dtype)
            w_right = torch.zeros(B, L, device=h_prev.device, dtype=h_prev.dtype)
            h_left  = torch.zeros_like(h_prev)
            h_right = torch.zeros_like(h_prev)

        w_self = self.self_w
        deg = (w_left + w_right + w_self).clamp_min(1e-6).unsqueeze(-1)  # (B,L,1)

        m = (w_left.unsqueeze(-1) * h_left +
             w_right.unsqueeze(-1) * h_right +
             w_self * h_prev) / deg

        x = self.node_enc(node_x)
        m = self.msg_enc(m)
        z = self.dropout(x + m)  # (B,L,H)

        zf = z.reshape(-1, H)
        hp = h_prev.reshape(-1, H)
        h_new = self.gru(zf, hp).reshape(B, L, H)
        return h_new

class TopoGraphGNN(nn.Module):
    """
    Causal spatio-temporal GNN with hierarchical link→motor coupling.

    Heads:
      - Link head: logits for (L+1)
      - Motor head (hierarchical):
          per-link 8-way logits from each node h_i, then add link logits of each link
          to form global motor logits of size (8L+1). Healthy(0) motor logit is tied to
          link-healthy logit plus a learnable bias.
    """
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden: int,
        gnn_layers: int,
        dropout: float,
        num_links: int,
        num_motors: int,  # 8L + 1
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_links = num_links
        self.num_motors = num_motors

        self.layers = nn.ModuleList(
            [GraphGRUCell(node_in if i == 0 else hidden, edge_in, hidden, self_loop_weight=1.0, dropout=dropout)
             for i in range(gnn_layers)]
        )

        # Link heads
        self.link_node_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),   # per-node faulty score
        )
        self.link_healthy_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),   # global healthy score
        )

        # Per-link motor head (8 logits per node)
        self.motor_cond_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, MOTORS_PER_LINK),
        )
        # Healthy motor bias, tied to link healthy logit
        self.motor_healthy_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_seq: torch.Tensor, edge_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        node_seq: (B, T, L, F_node)
        edge_seq: (B, T, L-1, F_edge)
        returns:
          link_logits:  (B, L+1)
          motor_logits: (B, 8L+1)  [hierarchical composition]
        """
        B, T, L, _ = node_seq.shape
        H = self.hidden
        h = torch.zeros(B, L, H, device=node_seq.device, dtype=node_seq.dtype)

        for t in range(T):
            x_t = node_seq[:, t]      # (B,L,F_node)
            e_t = edge_seq[:, t] if L > 1 else torch.zeros(B, 0, edge_seq.size(-1), device=node_seq.device, dtype=node_seq.dtype)
            for i, layer in enumerate(self.layers):
                h = layer(h, x_t if i == 0 else h, e_t)

        # ----- Link logits -----
        node_scores = self.link_node_head(h).squeeze(-1)            # (B,L)
        pooled = h.mean(dim=1)                                      # (B,H)
        healthy_score = self.link_healthy_head(pooled).squeeze(-1)  # (B,)
        link_logits = torch.cat([healthy_score.unsqueeze(-1), node_scores], dim=-1)  # (B,L+1)

        # ----- Hierarchical motor logits -----
        # Per-link conditional motor logits: (B, L, 8)
        cond_motor = self.motor_cond_head(h)

        # Add link logits (exclude healthy index 0) to their 8 motors
        # link_logits[:,1:] shape (B,L) -> (B,L,1) for broadcast
        combined = cond_motor + link_logits[:, 1:].unsqueeze(-1)  # (B,L,8)
        combined_flat = combined.reshape(B, L * MOTORS_PER_LINK)  # (B, 8L)

        # Healthy motor logit = link healthy + learnable bias
        healthy_motor_logit = link_logits[:, 0:1] + self.motor_healthy_bias  # (B,1)

        motor_logits = torch.cat([healthy_motor_logit, combined_flat], dim=1)  # (B, 8L+1)
        return link_logits, motor_logits

# ============================ Training helpers (AMP-ready) ============================

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def _hard_mask_motor_logits(motor_logits: torch.Tensor, link_pred: torch.Tensor, L: int) -> torch.Tensor:
    """
    Apply hard structural mask for motor logits at inference:
    - keep class 0 (healthy)
    - keep only the 8 motors of predicted link (1..L); others -> -inf
    motor_logits: (B, 8L+1), link_pred: (B,) in [0..L]
    """
    B = motor_logits.size(0)
    device = motor_logits.device
    # Build table: for each global motor class, which link it belongs to (0 for healthy)
    idx = torch.arange(0, L * MOTORS_PER_LINK + 1, device=device)
    global_link_id = torch.zeros_like(idx)                     # (8L+1,)
    if L > 0:
        global_link_id[1:] = ((idx[1:] - 1) // MOTORS_PER_LINK) + 1  # 1..L
    # allowed if same link OR class==0
    allowed = (global_link_id.unsqueeze(0) == link_pred.unsqueeze(1)) | (idx.unsqueeze(0) == 0)
    masked = torch.where(allowed, motor_logits, motor_logits.new_full(motor_logits.shape, -1e9))
    return masked

def _consistency_loss(link_logits: torch.Tensor, motor_logits: torch.Tensor, L: int) -> torch.Tensor:
    """
    KL(link || link_from_motor) + healthy consistency:
      - link_from_motor per link i = logsumexp over its 8 motor logits.
      - compare distributions over links (exclude healthy) with KL.
      - also align healthy probabilities (L1).
    """
    if L == 0:
        return motor_logits.new_zeros(())
    # Soft distributions
    p_link = F.softmax(link_logits[:, 1:], dim=1)                 # (B,L)
    # Aggregate motor logits back to link space
    B = motor_logits.size(0)
    link_from_motor_logits = motor_logits.new_zeros(B, L)
    for i in range(L):
        s = 1 + i * MOTORS_PER_LINK
        e = s + MOTORS_PER_LINK
        link_from_motor_logits[:, i] = torch.logsumexp(motor_logits[:, s:e], dim=1)
    log_q_link = F.log_softmax(link_from_motor_logits, dim=1)
    # KL: input is log-prob, target is prob
    kl = F.kl_div(log_q_link, p_link, reduction="batchmean")
    # Healthy consistency
    p_l0 = F.softmax(link_logits, dim=1)[:, 0]
    p_m0 = F.softmax(motor_logits, dim=1)[:, 0]
    healthy_l1 = F.l1_loss(p_m0, p_l0)
    return kl + healthy_l1

def run_epoch(
    model: TopoGraphGNN,
    loader: DataLoader,
    link_loss_fn: nn.Module,
    motor_loss_fn: nn.Module,
    motor_weight: float,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: Optional["torch.amp.GradScaler"],
    link_count: int,
    hard_mask_infer: bool,
    consistency_weight: float,
) -> Tuple[float, float, float]:
    from contextlib import nullcontext
    use_amp = (scaler is not None)
    autocast_cm = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if (use_amp and device.type == "cuda") else nullcontext()

    train = optimizer is not None
    model.train(mode=train)

    total_loss = 0.0
    total_link_correct = 0
    total_motor_correct = 0
    total_samples = 0

    for node_win, edge_win, link_targets, motor_targets in loader:
        node_win = node_win.to(device, non_blocking=True)   # (B,T,L,F_node)
        edge_win = edge_win.to(device, non_blocking=True)   # (B,T,L-1,F_edge)
        link_targets = link_targets.to(device, non_blocking=True)
        motor_targets = motor_targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast_cm:
            link_logits, motor_logits = model(node_win, edge_win)
            loss_link  = link_loss_fn(link_logits, link_targets)
            loss_motor = motor_loss_fn(motor_logits, motor_targets)
            loss = loss_link + motor_weight * loss_motor
            if consistency_weight > 0.0:
                loss = loss + consistency_weight * _consistency_loss(link_logits, motor_logits, link_count)

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        with torch.no_grad():
            bsz = link_targets.size(0)
            total_loss += loss.item() * bsz

            link_preds = link_logits.argmax(dim=1)  # (B,)
            # motor prediction: optionally hard-mask by predicted link for STRICT consistency at eval
            if (not train) and hard_mask_infer:
                masked_motor_logits = _hard_mask_motor_logits(motor_logits, link_preds, link_count)
                motor_preds = masked_motor_logits.argmax(dim=1)
            else:
                motor_preds = motor_logits.argmax(dim=1)

            total_link_correct += (link_preds == link_targets).sum().item()
            total_motor_correct += (motor_preds == motor_targets).sum().item()
            total_samples += bsz

    mean_loss = total_loss / max(total_samples, 1)
    link_acc  = total_link_correct / max(total_samples, 1)
    motor_acc = total_motor_correct / max(total_samples, 1)
    return mean_loss, link_acc, motor_acc

def split_train_val(
    node_features: np.ndarray,
    edge_features: np.ndarray,
    link_labels: np.ndarray,
    motor_labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    N = node_features.shape[0]
    perm = rng.permutation(N)
    val_count = max(1, int(N * val_ratio))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return (
        node_features[train_idx],
        edge_features[train_idx],
        link_labels[train_idx],
        motor_labels[train_idx],
        node_features[val_idx],
        edge_features[val_idx],
        link_labels[val_idx],
        motor_labels[val_idx],
    )

# ============================ CLI & main ============================

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LASDRA topological GNN fault detector (hierarchical constraints).")
    parser.add_argument("--data-root", type=str, default="data_storage/link_3", help="Shard directory.")
    parser.add_argument("--max-shards", type=int, default=0, help="Optional limit on number of shards to load (0=all).")
    parser.add_argument("--baseline-window", type=int, default=120, help="Frames used for baseline motor statistics.")
    parser.add_argument("--component-threshold-deg", type=float, default=18.0, help="Angle threshold for motor component count.")
    parser.add_argument("--per-motor-mode", choices=["full","none"], default="full",
                        help="Include per-motor 56-dim features (full) or drop them (none) for speed.")
    parser.add_argument("--component-metric", choices=["density","count"], default="density",
                        help="Motor topology summary: fast density (default) or exact component count (slow).")

    parser.add_argument("--window", type=int, default=96, help="Causal window length.")
    parser.add_argument("--pos-stride", type=int, default=3, help="Stride when the last frame is faulty.")
    parser.add_argument("--neg-stride", type=int, default=12, help="Stride when the last frame is healthy.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Holdout ratio by sequence.")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gnn-layers", type=int, default=1, help="Number of stacked GraphGRU layers.")

    parser.add_argument("--motor-weight", type=float, default=1.0, help="Relative weight for motor classification loss.")
    parser.add_argument("--consistency-weight", type=float, default=0.2, help="Weight for link↔motor consistency loss.")
    parser.add_argument("--hard-mask-infer", action="store_true", help="Apply hard structural mask for motor predictions at eval.")

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--save", type=str, default=None, help="Optional path to save best checkpoint.")

    # speed/engine toggles
    parser.add_argument("--amp", action="store_true", default=True, help="Use mixed precision on CUDA.")
    parser.add_argument("--compile", action="store_true", default=False, help="Use torch.compile if available.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    return parser

def main() -> None:
    args = parse_args().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")

    # backend perf knobs (TF32 on Ampere+)
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

    # ----- load & build features -----
    pack = load_all_shards(
        shard_paths,
        baseline_window=args.baseline_window,
        component_threshold_deg=args.component_threshold_deg,
        per_motor_mode=args.per_motor_mode,
        component_metric=args.component_metric,
    )
    node_all = pack.node_features  # (N,T,L,F_node)
    edge_all = pack.edge_features  # (N,T,L-1,F_edge)
    y_link_all  = pack.link_labels
    y_motor_all = pack.motor_labels
    L = pack.link_count
    print(
        f"[data] nodes {node_all.shape}, edges {edge_all.shape}, "
        f"link_labels {y_link_all.shape}, motor_labels {y_motor_all.shape} | links={L}"
    )

    # ----- split -----
    (node_tr, edge_tr, y_link_tr, y_motor_tr,
     node_val, edge_val, y_link_val, y_motor_val) = split_train_val(
        node_all, edge_all, y_link_all, y_motor_all, val_ratio=args.val_ratio, seed=args.seed
    )

    # ----- normalization -----
    n_mean, n_std, e_mean, e_std = compute_norm_stats_nodes_edges(node_tr, edge_tr)
    normalize_inplace_nodes_edges(node_tr, edge_tr, n_mean, n_std, e_mean, e_std)
    normalize_inplace_nodes_edges(node_val, edge_val, n_mean, n_std, e_mean, e_std)

    # ----- datasets / loaders -----
    train_ds = GraphWindowDataset(
        node_tr, edge_tr, y_link_tr, y_motor_tr,
        window=args.window, pos_stride=args.pos_stride, neg_stride=args.neg_stride
    )
    val_ds = GraphWindowDataset(
        node_val, edge_val, y_link_val, y_motor_val,
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
    val_loader   = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False, drop_last=False,
        num_workers=nw, pin_memory=pin, persistent_workers=pw, prefetch_factor=pf
    )

    # ----- model -----
    F_node = node_tr.shape[-1]
    F_edge = edge_tr.shape[-1]
    num_link_classes  = L + 1
    num_motor_classes = L * MOTORS_PER_LINK + 1

    model = TopoGraphGNN(
        node_in=F_node,
        edge_in=F_edge,
        hidden=args.hidden_size,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        num_links=L,
        num_motors=num_motor_classes,
    ).to(device)

    # optional compile (PyTorch 2.x) with safe fallback
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[warn] torch.compile failed ({e}); falling back to eager.", flush=True)

    # ----- losses & optimizer -----
    # You can add label smoothing here if desired: nn.CrossEntropyLoss(label_smoothing=0.05, weight=...)
    link_weights  = compute_class_weights(train_ds.link_labels,  num_link_classes).to(device)
    motor_weights = compute_class_weights(train_ds.motor_labels, num_motor_classes).to(device)
    link_loss_fn  = nn.CrossEntropyLoss(weight=link_weights)
    motor_loss_fn = nn.CrossEntropyLoss(weight=motor_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP scaler (new API with compatibility fallback)
    scaler = None
    if args.amp and device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            from torch.cuda.amp import GradScaler as CudaGradScaler
            scaler = CudaGradScaler(enabled=True)

    best_val_acc = 0.0
    best_epoch = 0

    # ----- train -----
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_link_acc, tr_motor_acc = run_epoch(
            model, train_loader, link_loss_fn, motor_loss_fn,
            args.motor_weight, optimizer, device, scaler,
            link_count=L, hard_mask_infer=False,  # do NOT hard-mask while training
            consistency_weight=args.consistency_weight,
        )
        val_loss, val_link_acc, val_motor_acc = run_epoch(
            model, val_loader, link_loss_fn, motor_loss_fn,
            args.motor_weight, optimizer=None, device=device, scaler=None,
            link_count=L, hard_mask_infer=args.hard_mask_infer,  # enable hard mask at eval if requested
            consistency_weight=args.consistency_weight,
        )
        scheduler.step()

        if val_link_acc > best_val_acc:
            best_val_acc = val_link_acc
            best_epoch = epoch
            if args.save:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "node_mean": n_mean,
                        "node_std":  n_std,
                        "edge_mean": e_mean,
                        "edge_std":  e_std,
                        "config": vars(args),
                        "val_link_acc": best_val_acc,
                        "link_count": L,
                        "node_dim": F_node,
                        "edge_dim": F_edge,
                    },
                    args.save,
                )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} train_link_acc={tr_link_acc*100:5.2f}% train_motor_acc={tr_motor_acc*100:5.2f}% | "
            f"val_loss={val_loss:.4f} val_link_acc={val_link_acc*100:5.2f}% val_motor_acc={val_motor_acc*100:5.2f}% | "
            f"best_link_acc={best_val_acc*100:5.2f}% (ep{best_epoch})"
        )

    print(f"[done] best validation link accuracy {best_val_acc*100:.2f}% at epoch {best_epoch}")

if __name__ == "__main__":
    main()
"""
python3 topo_gnn_fdi_train.py \
  --data-root data_storage/link_3 \
  --epochs 30 \
  --window 96 --pos-stride 3 --neg-stride 12 \
  --hidden-size 256 --batch-size 256 \
  --per-motor-mode full --component-metric density \
  --num-workers 8 --prefetch-factor 2 --persistent-workers \
  --amp \
  --consistency-weight 0.2 \
  --hard-mask-infer \
  --save topo_gnn.pt
"""
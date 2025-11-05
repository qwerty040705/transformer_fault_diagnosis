"""
Train a real-time LASDRA fault detector using topological SE(3) features.

This script scans all LASDRA shards, builds per-time-step feature tensors
based on se(3) residuals and motor-direction topology, and trains a causal
LSTM with an auxiliary fault-detection head.  It processes the entire
dataset (train/validation split by sequence) so the resulting model can be
 deployed for streaming inference.
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
from torch.utils.data import DataLoader, Dataset

MOTORS_PER_LINK = 8
EPS = 1e-9

# ------------------------- Feature extraction helpers -------------------------


def _unitize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, EPS)
    return vec / norm


def _so3_log_batch(R: np.ndarray) -> np.ndarray:
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


def compute_se3_residual(desired: np.ndarray, actual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos_des = desired[..., :3, 3]
    pos_act = actual[..., :3, 3]
    pos_err = pos_act - pos_des

    rot_des = desired[..., :3, :3]
    rot_act = actual[..., :3, :3]
    rot_err = np.matmul(np.swapaxes(rot_des, -1, -2), rot_act)
    omega = _so3_log_batch(rot_err)
    return omega.astype(np.float32, copy=False), pos_err.astype(np.float32, copy=False)


def default_motor_layout(link_count: int) -> np.ndarray:
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
    dirs = np.einsum("stlij,lmj->stlmi", rotations, motor_layout, optimize=True)
    return _unitize(dirs.astype(np.float32, copy=False))


def count_components(directions: np.ndarray, cos_thresh: float) -> int:
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


def build_link_targets(label_matrix: np.ndarray) -> np.ndarray:
    S, T, M = label_matrix.shape
    if M % MOTORS_PER_LINK != 0:
        raise ValueError(f"Label width {M} not divisible by {MOTORS_PER_LINK}.")
    L = M // MOTORS_PER_LINK
    reshaped = label_matrix.reshape(S, T, L, MOTORS_PER_LINK)
    fault_flags = reshaped.min(axis=-1) < 1
    targets = fault_flags.argmax(axis=-1) + 1
    healthy = ~fault_flags.any(axis=-1)
    targets[healthy] = 0
    return targets.astype(np.int64, copy=False)


def build_motor_targets(label_matrix: np.ndarray) -> np.ndarray:
    """
    Collapse motor-level labels into a single categorical target:
      0 → healthy, 1..(L*M) → faulty motor (global index).
    """
    S, T, M = label_matrix.shape
    faults = label_matrix < 1
    has_fault = faults.any(axis=-1)
    indices = np.argmax(faults, axis=-1) + 1  # provisional
    indices[~has_fault] = 0
    return indices.astype(np.int64, copy=False)


@dataclass
class FeaturePack:
    features: np.ndarray  # (S, T, L*F)
    link_labels: np.ndarray  # (S, T)
    motor_labels: np.ndarray  # (S, T)
    per_link_dim: int
    link_count: int


def build_topological_features(
    desired_cum: np.ndarray,
    actual_cum: np.ndarray,
    labels: np.ndarray,
    baseline_window: int,
    component_threshold_deg: float,
) -> FeaturePack:
    S, T, L, _, _ = desired_cum.shape
    omega, pos_err = compute_se3_residual(desired_cum, actual_cum)
    rotations = actual_cum[..., :3, :3]
    motor_layout = default_motor_layout(L)
    motor_dirs = compute_motor_directions(rotations, motor_layout)  # (S,T,L,M,3)

    cos_component = math.cos(math.radians(component_threshold_deg))

    feature_list: List[np.ndarray] = []

    for s in range(S):
        seq_features = []
        seq_motors = motor_dirs[s]  # (T,L,M,3)
        base_window = min(baseline_window, T)
        if base_window <= 0:
            raise ValueError("baseline_window must be positive.")
        baseline_dirs = _unitize(seq_motors[:base_window].mean(axis=0, keepdims=False))  # (L,M,3)

        for l in range(L):
            omega_l = omega[s, :, l, :]  # (T,3)
            pos_l = pos_err[s, :, l, :]
            energy_l = np.sum(omega_l * omega_l + pos_l * pos_l, axis=-1, keepdims=True)  # (T,1)
            motor_l = seq_motors[:, l, :, :]  # (T,M,3)
            baseline_l = baseline_dirs[l]

            diff = motor_l - baseline_l[None, :, :]  # (T,M,3)
            dots = np.clip(np.sum(motor_l * baseline_l[None, :, :], axis=-1), -1.0, 1.0)
            angles = np.arccos(dots)[..., None]  # (T,M,1)
            per_motor = np.concatenate([motor_l, diff, angles], axis=-1).reshape(T, -1)  # (T, M*7)

            components = np.empty((T, 1), dtype=np.float32)
            for t in range(T):
                components[t, 0] = count_components(motor_l[t], cos_component) / MOTORS_PER_LINK

            link_feats = np.concatenate(
                [
                    omega_l,
                    pos_l,
                    energy_l,
                    components,
                    per_motor,
                ],
                axis=-1,
            )  # (T, F_link)

            seq_features.append(link_feats)

        seq_features = np.concatenate(seq_features, axis=-1)  # (T, L*F_link)
        feature_list.append(seq_features[None, :, :])

    features = np.concatenate(feature_list, axis=0).astype(np.float32, copy=False)
    link_labels = build_link_targets(labels)
    motor_labels = build_motor_targets(labels)
    per_link_dim = features.shape[-1] // L
    return FeaturePack(
        features=features,
        link_labels=link_labels,
        motor_labels=motor_labels,
        per_link_dim=per_link_dim,
        link_count=L,
    )


def discover_shards(root: str) -> List[Path]:
    pattern = str(Path(root) / "fault_dataset_shard_*.npz")
    return [Path(p) for p in sorted(glob.glob(pattern))]


def load_all_shards(
    paths: Sequence[Path],
    baseline_window: int,
    component_threshold_deg: float,
) -> FeaturePack:
    feat_buf: List[np.ndarray] = []
    link_buf: List[np.ndarray] = []
    motor_buf: List[np.ndarray] = []
    per_link_dim = None
    link_count = None

    for path in paths:
        with np.load(path) as data:
            desired_cum = data["desired_link_cum"]
            actual_cum = data["actual_link_cum"]
            labels = data["label"]

        pack = build_topological_features(
            desired_cum,
            actual_cum,
            labels,
            baseline_window=baseline_window,
            component_threshold_deg=component_threshold_deg,
        )
        feat_buf.append(pack.features)
        link_buf.append(pack.link_labels)
        motor_buf.append(pack.motor_labels)

        if per_link_dim is None:
            per_link_dim = pack.per_link_dim
            link_count = pack.link_count
        else:
            if per_link_dim != pack.per_link_dim:
                raise ValueError("Inconsistent per-link feature dimension across shards.")
            if link_count != pack.link_count:
                raise ValueError("Inconsistent link count across shards.")

    features = np.concatenate(feat_buf, axis=0)
    link_labels = np.concatenate(link_buf, axis=0)
    motor_labels = np.concatenate(motor_buf, axis=0)
    return FeaturePack(
        features=features,
        link_labels=link_labels,
        motor_labels=motor_labels,
        per_link_dim=per_link_dim,
        link_count=link_count or 0,
    )


# ----------------------------- Dataset utilities ------------------------------


class TopoWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        link_labels: np.ndarray,
        motor_labels: np.ndarray,
        window: int,
        pos_stride: int,
        neg_stride: int,
    ) -> None:
        if features.ndim != 3:
            raise ValueError("features must be (N, T, F).")
        if link_labels.shape[:2] != features.shape[:2]:
            raise ValueError("link_labels shape mismatch.")
        if motor_labels.shape[:2] != features.shape[:2]:
            raise ValueError("motor_labels shape mismatch.")
        if window <= 0 or window > features.shape[1]:
            raise ValueError("invalid window length.")

        self.features = features
        self.link_full = link_labels
        self.motor_full = motor_labels
        self.window = window

        slots_seq: List[int] = []
        slots_start: List[int] = []
        slots_link: List[int] = []
        slots_motor: List[int] = []

        N, T, _ = features.shape
        for seq in range(N):
            end = window
            while end <= T:
                link_y = int(link_labels[seq, end - 1])
                motor_y = int(motor_labels[seq, end - 1])
                start = end - window
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.seq_idx[idx]
        start = self.start_idx[idx]
        end = start + self.window
        window = self.features[seq, start:end]
        link_label = self.link_labels[idx]
        motor_label = self.motor_labels[idx]
        return (
            torch.from_numpy(window),
            torch.tensor(link_label, dtype=torch.int64),
            torch.tensor(motor_label, dtype=torch.int64),
        )


def compute_norm_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_inplace(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    features -= mean
    features /= std


# ------------------------------- Model definition -----------------------------


class TopoLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_link_classes: int,
        num_motor_classes: int,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.link_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_link_classes),
        )
        self.motor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_motor_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_out, _ = self.lstm(x)
        last = seq_out[:, -1, :]
        link_logits = self.link_head(last)
        motor_logits = self.motor_head(last)
        return link_logits, motor_logits


# ------------------------------ Training helpers ------------------------------


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: TopoLSTM,
    loader: DataLoader,
    link_loss_fn: nn.Module,
    motor_loss_fn: nn.Module,
    motor_weight: float,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float, float]:
    train = optimizer is not None
    model.train(mode=train)

    total_loss = 0.0
    total_link_correct = 0
    total_motor_correct = 0
    total_samples = 0

    for inputs, link_targets, motor_targets in loader:
        inputs = inputs.to(device)
        link_targets = link_targets.to(device)
        motor_targets = motor_targets.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        link_logits, motor_logits = model(inputs)
        loss_link = link_loss_fn(link_logits, link_targets)
        loss_motor = motor_loss_fn(motor_logits, motor_targets)
        loss = loss_link + motor_weight * loss_motor

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = link_targets.size(0)
        total_loss += loss.item() * batch_size

        link_preds = link_logits.argmax(dim=1)
        motor_preds = motor_logits.argmax(dim=1)
        total_link_correct += (link_preds == link_targets).sum().item()
        total_motor_correct += (motor_preds == motor_targets).sum().item()
        total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    link_acc = total_link_correct / max(total_samples, 1)
    motor_acc = total_motor_correct / max(total_samples, 1)
    return mean_loss, link_acc, motor_acc


def split_train_val(
    features: np.ndarray,
    link_labels: np.ndarray,
    motor_labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    N = features.shape[0]
    perm = rng.permutation(N)
    val_count = max(1, int(N * val_ratio))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return (
        features[train_idx],
        link_labels[train_idx],
        motor_labels[train_idx],
        features[val_idx],
        link_labels[val_idx],
        motor_labels[val_idx],
    )


# ----------------------------------- CLI --------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LASDRA topological fault detector.")
    parser.add_argument("--data-root", type=str, default="data_storage/link_3", help="Shard directory.")
    parser.add_argument("--max-shards", type=int, default=0, help="Optional limit on number of shards to load (0=all).")
    parser.add_argument("--baseline-window", type=int, default=120, help="Frames used for baseline motor statistics.")
    parser.add_argument("--component-threshold-deg", type=float, default=18.0, help="Angle threshold for component count.")
    parser.add_argument("--window", type=int, default=96, help="Causal window length.")
    parser.add_argument("--pos-stride", type=int, default=3, help="Stride for fault windows.")
    parser.add_argument("--neg-stride", type=int, default=12, help="Stride for healthy windows.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Holdout ratio by sequence.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--motor-weight", type=float, default=1.0, help="Relative weight for motor classification loss.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--save", type=str, default=None, help="Optional path to save best checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")

    shard_paths = discover_shards(args.data_root)
    if not shard_paths:
        raise FileNotFoundError(f"No shards found at {args.data_root}")
    if args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]
    print(f"[data] found {len(shard_paths)} shards")

    pack = load_all_shards(
        shard_paths,
        baseline_window=args.baseline_window,
        component_threshold_deg=args.component_threshold_deg,
    )
    features = pack.features
    link_labels = pack.link_labels
    motor_labels = pack.motor_labels
    print(f"[data] features shape {features.shape}, link labels shape {link_labels.shape}, motor labels shape {motor_labels.shape}")

    X_train, y_link_train, y_motor_train, X_val, y_link_val, y_motor_val = split_train_val(
        features,
        link_labels,
        motor_labels,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    mean, std = compute_norm_stats(X_train)
    normalize_inplace(X_train, mean, std)
    normalize_inplace(X_val, mean, std)

    train_ds = TopoWindowDataset(
        X_train,
        y_link_train,
        y_motor_train,
        window=args.window,
        pos_stride=args.pos_stride,
        neg_stride=args.neg_stride,
    )
    val_ds = TopoWindowDataset(
        X_val,
        y_link_val,
        y_motor_val,
        window=args.window,
        pos_stride=args.pos_stride,
        neg_stride=args.neg_stride,
    )
    print(f"[data] windows train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=device.type == "cuda")

    num_features = X_train.shape[-1]
    num_link_classes = pack.link_count + 1
    num_motor_classes = pack.link_count * MOTORS_PER_LINK + 1

    model = TopoLSTM(
        input_dim=num_features,
        hidden_size=args.hidden_size,
        num_layers=args.lstm_layers,
        dropout=args.dropout,
        num_link_classes=num_link_classes,
        num_motor_classes=num_motor_classes,
    ).to(device)

    link_weights = compute_class_weights(train_ds.link_labels, num_link_classes).to(device)
    motor_weights = compute_class_weights(train_ds.motor_labels, num_motor_classes).to(device)
    link_loss_fn = nn.CrossEntropyLoss(weight=link_weights)
    motor_loss_fn = nn.CrossEntropyLoss(weight=motor_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_link_acc, train_motor_acc = run_epoch(
            model,
            train_loader,
            link_loss_fn,
            motor_loss_fn,
            args.motor_weight,
            optimizer,
            device,
        )
        val_loss, val_link_acc, val_motor_acc = run_epoch(
            model,
            val_loader,
            link_loss_fn,
            motor_loss_fn,
            args.motor_weight,
            optimizer=None,
            device=device,
        )
        scheduler.step()

        if val_link_acc > best_val_acc:
            best_val_acc = val_link_acc
            best_epoch = epoch
            if args.save:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "mean": mean,
                        "std": std,
                        "config": vars(args),
                        "val_link_acc": best_val_acc,
                        "per_link_dim": pack.per_link_dim,
                        "link_count": pack.link_count,
                    },
                    args.save,
                )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_link_acc={train_link_acc*100:5.2f}% train_motor_acc={train_motor_acc*100:5.2f}% | "
            f"val_loss={val_loss:.4f} val_link_acc={val_link_acc*100:5.2f}% val_motor_acc={val_motor_acc*100:5.2f}% | "
            f"best_link_acc={best_val_acc*100:5.2f}% (ep{best_epoch})"
        )

    print(f"[done] best validation link accuracy {best_val_acc*100:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()

"""
python3 topo_fdi_train.py \
  --epochs 30 \
  --baseline-window 120 \
  --component-threshold-deg 18 \
  --window 96 \
  --pos-stride 3 \
  --neg-stride 12 \
  --batch-size 256 \
  --val-batch-size 512 \
  --val-ratio 0.2 \
  --motor-weight 1.0 \
  --save topo_lstm.pt
"""
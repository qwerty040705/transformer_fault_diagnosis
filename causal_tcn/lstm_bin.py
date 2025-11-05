"""
Causal link-fault identifier for a 3-link chain.

This script trains a causal LSTM to detect faults in real time and to
identify which link is affected.  It constructs rich
per-link feature tensors (position/orientation residuals, velocities,
accelerations, and command/response mismatches) from the dataset shards
in ``data_storage/link_3/`` and evaluates the model on held-out shards.

Expected usage::

    python causal_tcn/lstm_bin.py

To customise hyper-parameters (window length, hidden width, etc.), see
the argument parser in :func:`main`.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ----------------------------- Hyperparameters -----------------------------
DEFAULT_DATA_ROOT = os.path.join("data_storage", "link_3")
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_VAL_BATCH = 512
DEFAULT_WINDOW = 96
DEFAULT_POS_STRIDE = 3
DEFAULT_NEG_STRIDE = 12
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.1
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_LSTM_LAYERS = 2
DEFAULT_FAULT_LOSS_WEIGHT = 0.5
DEFAULT_VAL_SHARDS = 2
MOTORS_PER_LINK = 8
EPS_STD = 1e-5


# ----------------------------- Utility helpers -----------------------------
def set_all_seeds(seed: int) -> None:
    """Fix random seeds for reproducible training trails."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def discover_shards(root: str) -> List[str]:
    pattern = os.path.join(root, "fault_dataset_shard_*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No dataset shards found at: {pattern}")
    return paths


def _so3_log_batch(Rm: np.ndarray) -> np.ndarray:
    """Compute log map of SO(3) matrices → axis-angle vectors."""
    trace = np.clip((np.trace(Rm, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(trace)
    sin_theta = np.sin(theta)

    skew = Rm - np.swapaxes(Rm, -1, -2)
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


def build_pose_error(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Compute per-link pose residuals between desired and actual transforms.

    Returns an array of shape (S, T, L, 6) storing translation and
    orientation errors.
    """
    pos_des = desired[..., :3, 3]
    pos_act = actual[..., :3, 3]
    pos_err = pos_act - pos_des

    rot_des = desired[..., :3, :3]
    rot_act = actual[..., :3, :3]
    rot_err = np.matmul(np.swapaxes(rot_des, -1, -2), rot_act)
    rot_vec = _so3_log_batch(rot_err)
    return np.concatenate([pos_err, rot_vec], axis=-1).astype(np.float32, copy=False)


def temporal_derivative(signal: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute a causal finite difference derivative along the time axis.

    The derivative is zero-padded at the first time-step to preserve shape.
    """
    deriv = np.zeros_like(signal, dtype=np.float32)
    deriv[:, 1:] = (signal[:, 1:] - signal[:, :-1]) / dt
    return deriv


def compute_pose_velocity(transforms: np.ndarray, dt: float) -> np.ndarray:
    """
    Estimate per-link spatial velocity from SE(3) transforms.

    Translation velocity is computed via forward differences, and the
    rotational component uses the log map of successive rotation deltas.
    """
    pos = transforms[..., :3, 3]
    vel_pos = np.zeros_like(pos, dtype=np.float32)
    vel_pos[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / dt

    rot = transforms[..., :3, :3]
    rot_prev = rot[:, :-1]
    rot_next = rot[:, 1:]
    rel = np.matmul(np.swapaxes(rot_prev, -1, -2), rot_next)
    vel_rot = np.zeros_like(vel_pos, dtype=np.float32)
    vel_rot[:, 1:] = _so3_log_batch(rel) / dt

    return np.concatenate([vel_pos, vel_rot], axis=-1).astype(np.float32, copy=False)


def build_feature_stack(
    desired_cum: np.ndarray,
    actual_cum: np.ndarray,
    desired_rel: np.ndarray,
    actual_rel: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Assemble the full per-link feature bank as discussed with the user."""
    dt = float(dt)

    # Base residuals
    abs_err = build_pose_error(desired_cum, actual_cum)
    rel_err = build_pose_error(desired_rel, actual_rel)

    # Residual dynamics (velocity & acceleration)
    abs_err_vel = temporal_derivative(abs_err, dt)
    abs_err_acc = temporal_derivative(abs_err_vel, dt)
    rel_err_vel = temporal_derivative(rel_err, dt)
    rel_err_acc = temporal_derivative(rel_err_vel, dt)

    # Commanded / actual velocities and accelerations (absolute frame)
    act_vel_abs = compute_pose_velocity(actual_cum, dt)
    act_acc_abs = temporal_derivative(act_vel_abs, dt)
    des_vel_abs = compute_pose_velocity(desired_cum, dt)
    des_acc_abs = temporal_derivative(des_vel_abs, dt)

    # Commanded / actual velocities and accelerations (relative frame)
    act_vel_rel = compute_pose_velocity(actual_rel, dt)
    act_acc_rel = temporal_derivative(act_vel_rel, dt)
    des_vel_rel = compute_pose_velocity(desired_rel, dt)
    des_acc_rel = temporal_derivative(des_vel_rel, dt)

    # Velocity / acceleration mismatches
    vel_diff_abs = act_vel_abs - des_vel_abs
    acc_diff_abs = act_acc_abs - des_acc_abs
    vel_diff_rel = act_vel_rel - des_vel_rel
    acc_diff_rel = act_acc_rel - des_acc_rel

    feature_list = [
        abs_err,
        rel_err,
        abs_err_vel,
        abs_err_acc,
        rel_err_vel,
        rel_err_acc,
        act_vel_abs,
        act_acc_abs,
        des_vel_abs,
        des_acc_abs,
        act_vel_rel,
        act_acc_rel,
        des_vel_rel,
        des_acc_rel,
        vel_diff_abs,
        acc_diff_abs,
        vel_diff_rel,
        acc_diff_rel,
    ]
    stacked = np.concatenate(feature_list, axis=-1)
    return stacked.astype(np.float32, copy=False)


def build_link_targets(label_matrix: np.ndarray) -> np.ndarray:
    """
    Collapse 8-motor labels per link into a single categorical target:
      0 → healthy (no fault), 1..L → faulty link index (1-based).
    """
    S, T, M = label_matrix.shape
    if M % MOTORS_PER_LINK != 0:
        raise ValueError(f"Label width {M} is not divisible by {MOTORS_PER_LINK}.")
    L = M // MOTORS_PER_LINK
    reshaped = label_matrix.reshape(S, T, L, MOTORS_PER_LINK)
    fault_flags = reshaped.min(axis=-1) < 1

    targets = fault_flags.argmax(axis=-1) + 1  # provisional (1..L)
    healthy_mask = ~fault_flags.any(axis=-1)
    targets[healthy_mask] = 0
    return targets.astype(np.int64, copy=False)


def load_shards(paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load and stack features/targets from a list of shard paths."""
    feature_buf = []
    target_buf = []
    link_count = None
    per_link_dim = None
    dt_ref = None

    for path in paths:
        with np.load(path) as data:
            desired_cum = data["desired_link_cum"]  # (S, T, L, 4, 4)
            actual_cum = data["actual_link_cum"]
            desired_rel = data["desired_link_rel"]
            actual_rel = data["actual_link_rel"]
            labels = data["label"]
            dt = float(data["dt"])
            link_cnt = int(data["link_count"])

        if dt_ref is None:
            dt_ref = dt
        elif not np.isclose(dt_ref, dt):
            raise ValueError(f"Inconsistent dt detected ({dt_ref} vs {dt}) in {path}")

        if link_count is None:
            link_count = link_cnt
        elif link_count != link_cnt:
            raise ValueError(f"Inconsistent link_count detected ({link_count} vs {link_cnt}) in {path}")

        feats = build_feature_stack(desired_cum, actual_cum, desired_rel, actual_rel, dt)  # (S, T, L, F)
        S, T, L, F = feats.shape
        if per_link_dim is None:
            per_link_dim = F
        elif per_link_dim != F:
            raise ValueError(f"Inconsistent per-link feature width ({per_link_dim} vs {F}).")

        targets = build_link_targets(labels)  # (S, T)
        feature_buf.append(feats.reshape(S, T, L * F))
        target_buf.append(targets)

    features = np.concatenate(feature_buf, axis=0)
    targets = np.concatenate(target_buf, axis=0)
    return features, targets, int(link_count), int(per_link_dim)


def compute_norm_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < EPS_STD] = EPS_STD
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_inplace(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    features -= mean
    features /= std


# ----------------------------- Dataset wrapper -----------------------------
class FaultWindowDataset(Dataset):
    """
    Generates causal windows from per-sequence features.
    Each sample is shaped (window, feature_dim) with label ∈ {0,1,..,L}.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window: int,
        pos_stride: int,
        neg_stride: int,
    ) -> None:
        if features.ndim != 3:
            raise ValueError("features must be (N, T, F).")
        if targets.shape[:2] != features.shape[:2]:
            raise ValueError("targets shape mismatch.")
        if window <= 0 or window > features.shape[1]:
            raise ValueError(f"Invalid window length: {window}")
        if pos_stride <= 0 or neg_stride <= 0:
            raise ValueError("Strides must be positive.")

        self.features = features
        self.targets = targets
        self.window = window
        self.total_seq, self.seq_len, self.feat_dim = features.shape

        seq_slots: List[int] = []
        start_slots: List[int] = []
        label_slots: List[int] = []

        for seq_idx in range(self.total_seq):
            end = window
            while end <= self.seq_len:
                label = int(targets[seq_idx, end - 1])
                start = end - window
                seq_slots.append(seq_idx)
                start_slots.append(start)
                label_slots.append(label)
                step = pos_stride if label != 0 else neg_stride
                end += step

        self.seq_idx = np.asarray(seq_slots, dtype=np.int32)
        self.start_idx = np.asarray(start_slots, dtype=np.int32)
        self.labels = np.asarray(label_slots, dtype=np.int64)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.seq_idx[idx]
        start = self.start_idx[idx]
        end = start + self.window
        window = self.features[seq, start:end]  # (window, feat_dim)
        label = self.labels[idx]
        x = torch.from_numpy(window)
        y = torch.tensor(label, dtype=torch.int64)
        return x, y


# ----------------------------- Model definition -----------------------------
class LSTMClassifier(nn.Module):
    """Causal LSTM encoder with dual output heads."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        fused_dim = hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.detector = nn.Sequential(
            nn.Linear(fused_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_out, _ = self.lstm(x)
        last = seq_out[:, -1, :]
        class_logits = self.classifier(last)
        fault_logits = self.detector(last).squeeze(-1)
        return class_logits, fault_logits


# ----------------------------- Training utilities -----------------------------
def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def compute_fault_pos_weight(labels: np.ndarray) -> float:
    healthy = float((labels == 0).sum())
    faults = float((labels != 0).sum())
    if faults <= 0.0:
        return 1.0
    return healthy / faults


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_criterion: nn.Module,
    fault_criterion: nn.Module,
    fault_loss_weight: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            class_logits, fault_logits = model(inputs)
            ce_loss = ce_criterion(class_logits, targets)
            fault_targets = (targets != 0).float()
            fault_loss = fault_criterion(fault_logits, fault_targets)
            loss = ce_loss + fault_loss_weight * fault_loss

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        preds = class_logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    mean_loss = total_loss / max(total_samples, 1)
    mean_acc = total_correct / max(total_samples, 1)
    return mean_loss, mean_acc


# ----------------------------- Main entry point -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train causal link fault detector (LSTM).")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="Path to shard directory.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--val-batch-size", type=int, default=DEFAULT_VAL_BATCH, help="Validation batch size.")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Causal window length.")
    parser.add_argument("--pos-stride", type=int, default=DEFAULT_POS_STRIDE, help="Stride after fault onset.")
    parser.add_argument("--neg-stride", type=int, default=DEFAULT_NEG_STRIDE, help="Stride while healthy.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout probability.")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE, help="LSTM hidden size.")
    parser.add_argument("--lstm-layers", type=int, default=DEFAULT_LSTM_LAYERS, help="Number of LSTM layers.")
    parser.add_argument(
        "--fault-loss-weight",
        type=float,
        default=DEFAULT_FAULT_LOSS_WEIGHT,
        help="Weighting factor for the binary fault-detection loss.",
    )
    parser.add_argument("--val-shards", type=int, default=DEFAULT_VAL_SHARDS, help="Number of shards for validation.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save best model weights (.pt).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] using {device}")

    shard_paths = discover_shards(args.data_root)
    val_shards = max(1, min(args.val_shards, len(shard_paths) // 5 or 1))
    train_paths = shard_paths[:-val_shards] or shard_paths
    val_paths = shard_paths[-val_shards:] if len(shard_paths) > val_shards else shard_paths

    print(f"[data] train shards: {len(train_paths)}  |  val shards: {len(val_paths)}")

    train_features, train_targets, link_count, per_link_dim = load_shards(train_paths)
    val_features, val_targets, link_count_val, per_link_dim_val = load_shards(val_paths)

    if link_count_val != link_count or per_link_dim_val != per_link_dim:
        raise ValueError("Validation shards reported inconsistent link metadata.")

    mean, std = compute_norm_stats(train_features)
    normalize_inplace(train_features, mean, std)
    normalize_inplace(val_features, mean, std)

    dataset_train = FaultWindowDataset(
        train_features,
        train_targets,
        window=args.window,
        pos_stride=args.pos_stride,
        neg_stride=args.neg_stride,
    )
    dataset_val = FaultWindowDataset(
        val_features,
        val_targets,
        window=args.window,
        pos_stride=args.pos_stride,
        neg_stride=args.neg_stride,
    )

    print(f"[data] train samples: {len(dataset_train)}  |  val samples: {len(dataset_val)}")

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    num_features = train_features.shape[-1]
    num_classes = link_count + 1

    model = LSTMClassifier(
        in_features=num_features,
        hidden_size=args.hidden_size,
        num_layers=args.lstm_layers,
        dropout=args.dropout,
        num_classes=num_classes,
    ).to(device)

    class_weights = compute_class_weights(dataset_train.labels, num_classes).to(device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

    fault_pos_weight = compute_fault_pos_weight(dataset_train.labels)
    fault_weight_tensor = torch.tensor([fault_pos_weight], dtype=torch.float32, device=device)
    fault_criterion = nn.BCEWithLogitsLoss(pos_weight=fault_weight_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            ce_criterion,
            fault_criterion,
            args.fault_loss_weight,
            optimizer,
            device,
            train=True,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            ce_criterion,
            fault_criterion,
            args.fault_loss_weight,
            optimizer,
            device,
            train=False,
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            if args.save:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "mean": mean,
                        "std": std,
                        "config": vars(args),
                        "val_acc": best_val_acc,
                        "link_count": link_count,
                        "per_link_dim": per_link_dim,
                    },
                    args.save,
                )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:5.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:5.2f}% | "
            f"best={best_val_acc*100:5.2f}% (ep{best_epoch})"
        )

    print(f"[done] best validation accuracy {best_val_acc*100:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()

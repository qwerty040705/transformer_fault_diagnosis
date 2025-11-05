"""
LASDRA Fault Topology Learning pipeline.

This module ingests LASDRA SE(3) shard files and extracts the topological
fault descriptors discussed in the planning notes:

* se(3) residual energy per link after the fault onset.
* Connected-component shifts of motor direction trajectories on S^2.
* Geodesic deviation energy and Wasserstein distance for motor-level IDs.

The script prints per-sequence diagnostics and highlights the predicted
faulty link and the dominant faulty motor according to the geometric
criteria.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

MOTORS_PER_LINK = 8
EPS = 1e-9


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
    """
    Compute se(3) residuals between desired and actual cumulative transforms.

    Returns:
        omega: (S, T, L, 3) axis-angle residual vectors.
        pos_err: (S, T, L, 3) translation residuals.
    """
    pos_des = desired[..., :3, 3]
    pos_act = actual[..., :3, 3]
    pos_err = pos_act - pos_des

    rot_des = desired[..., :3, :3]
    rot_act = actual[..., :3, :3]
    rot_err = np.matmul(np.swapaxes(rot_des, -1, -2), rot_act)
    omega = _so3_log_batch(rot_err)
    return omega.astype(np.float32, copy=False), pos_err.astype(np.float32, copy=False)


def default_motor_layout(link_count: int, motors_per_link: int = MOTORS_PER_LINK) -> np.ndarray:
    """Return a simple evenly spread set of local motor directions."""
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
    base = _unitize(base)
    if base.shape[0] != motors_per_link:
        raise ValueError("Default layout expects exactly 8 motors per link.")
    layout = np.tile(base[None, :, :], (link_count, 1, 1))
    return layout


def compute_motor_directions(rotations: np.ndarray, motor_layout: np.ndarray) -> np.ndarray:
    """
    Map local motor directions through the actual link rotation matrices.

    Args:
        rotations: (S, T, L, 3, 3) actual rotation matrices.
        motor_layout: (L, M, 3) nominal local directions.
    Returns:
        dirs: (S, T, L, M, 3) unit vectors in world frame.
    """
    dirs = np.einsum("stlij,lmj->stlmi", rotations, motor_layout, optimize=True)
    return _unitize(dirs.astype(np.float32, copy=False))


def fault_onset_from_labels(labels: np.ndarray, link_index: int) -> Optional[int]:
    """
    Return the first time index where any motor in link 'link_index' becomes faulty.
    Labels are expected to be binary with shape (T, L*8).
    """
    l0 = link_index * MOTORS_PER_LINK
    l1 = l0 + MOTORS_PER_LINK
    healthy = labels[:, l0:l1].min(axis=-1) > 0.5
    fault_idx = np.where(~healthy)[0]
    if fault_idx.size == 0:
        return None
    return int(fault_idx[0])


def integrate_residual_energy(omega: np.ndarray, pos_err: np.ndarray, dt: float, start: Optional[int]) -> float:
    """Integrate se(3) residual energy from 'start' to the end of the trajectory."""
    if start is None:
        return 0.0
    if start >= omega.shape[0]:
        return 0.0
    slice_o = omega[start:]
    slice_p = pos_err[start:]
    integrand = np.sum(slice_o * slice_o + slice_p * slice_p, axis=-1)
    return float(np.sum(integrand) * dt)


def spherical_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)
    return np.arccos(dot)


def count_connected_components(points: np.ndarray, threshold_rad: float) -> int:
    """
    Count connected components on the sphere using an angular threshold for adjacency.
    """
    if points.shape[0] == 0:
        return 0
    cos_thresh = np.cos(threshold_rad)
    visited = np.zeros(points.shape[0], dtype=bool)
    components = 0
    for i in range(points.shape[0]):
        if visited[i]:
            continue
        components += 1
        queue = [i]
        visited[i] = True
        while queue:
            idx = queue.pop()
            dots = np.sum(points[idx] * points, axis=-1)
            neighbors = np.where((dots >= cos_thresh) & (~visited))[0]
            if neighbors.size == 0:
                continue
            visited[neighbors] = True
            queue.extend(neighbors.tolist())
    return components


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Approximate the W2 distance between two 1D empirical distributions.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    n = max(a.size, b.size)
    grid = (np.arange(n) + 0.5) / n
    qa = np.interp(grid, (np.arange(a.size) + 0.5) / a.size, np.sort(a))
    qb = np.interp(grid, (np.arange(b.size) + 0.5) / b.size, np.sort(b))
    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def geodesic_deviation_energy(fault_dirs: np.ndarray, baseline: np.ndarray, dt: float) -> float:
    """
    Integrate squared spherical distance between fault directions and baseline.
    """
    if fault_dirs.shape[0] == 0:
        return 0.0
    dists = spherical_distance(fault_dirs, baseline[None, :])
    return float(np.sum(dists * dists) * dt)


@dataclass
class MotorMetrics:
    motor_index: int
    geodesic_energy: float
    wasserstein: float


@dataclass
class LinkMetrics:
    link_index: int
    onset_index: Optional[int]
    residual_energy: float
    component_shift: int
    detection_index: Optional[int]
    detection_time: Optional[float]
    detection_delay: Optional[float]
    dominant_motor: Optional[int]
    dominant_motor_energy: float
    dominant_motor_wasserstein: float


@dataclass
class SequenceReport:
    sample_index: int
    predicted_link: Optional[int]
    predicted_motor: Optional[int]
    true_link: Optional[int]
    true_motor: Optional[int]
    link_metrics: List[LinkMetrics]


class TopologicalFDIPipeline:
    def __init__(
        self,
        component_threshold_deg: float = 15.0,
        motor_layout: Optional[np.ndarray] = None,
        baseline_window: int = 100,
        std_scale: float = 4.0,
        motor_window: int = 30,
        motor_forward_window: int = 20,
        amplitude_scale: float = 12.0,
    ) -> None:
        self.component_threshold_rad = np.deg2rad(component_threshold_deg)
        self.motor_layout = motor_layout
        self.baseline_window = baseline_window
        self.std_scale = std_scale
        self.motor_window = motor_window
        self.motor_forward_window = motor_forward_window
        self.amplitude_scale = amplitude_scale

    def _ensure_layout(self, link_count: int) -> np.ndarray:
        if self.motor_layout is None:
            self.motor_layout = default_motor_layout(link_count)
        if self.motor_layout.shape[0] != link_count:
            raise ValueError("Motor layout link dimension mismatch.")
        return self.motor_layout

    def evaluate_sequence(
        self,
        sample_idx: int,
        omega: np.ndarray,
        pos_err: np.ndarray,
        labels: np.ndarray,
        motor_dirs: np.ndarray,
        dt: float,
    ) -> SequenceReport:
        T, L, _ = omega.shape
        link_metrics: List[LinkMetrics] = []
        detection_candidates: List[Tuple[float, float, int]] = []

        true_link = None
        true_onset = None

        for link in range(L):
            onset = fault_onset_from_labels(labels, link)
            split = onset if onset is not None else T
            residual_energy = integrate_residual_energy(omega[:, link, :], pos_err[:, link, :], dt, onset)

            energy_series = np.sum(
                omega[:, link, :] * omega[:, link, :] + pos_err[:, link, :] * pos_err[:, link, :],
                axis=-1,
            )
            detection_idx = self._detect_from_energy(energy_series)
            detection_time = detection_idx * dt if detection_idx is not None else None
            detection_delay = None
            if detection_idx is not None and onset is not None:
                detection_delay = (detection_idx - onset) * dt

            healthy_dirs = motor_dirs[:split, link].reshape(-1, motor_dirs.shape[-1])
            fault_dirs = motor_dirs[split:, link].reshape(-1, motor_dirs.shape[-1])
            healthy_components = count_connected_components(healthy_dirs, self.component_threshold_rad)
            fault_components = count_connected_components(fault_dirs, self.component_threshold_rad)
            component_shift = fault_components - healthy_components

            motor_metrics = self._evaluate_motors(motor_dirs[:, link], onset, detection_idx, dt)
            if motor_metrics:
                dominant = max(motor_metrics, key=lambda m: m.geodesic_energy)
                dom_idx = dominant.motor_index
                dom_energy = dominant.geodesic_energy
                dom_wass = dominant.wasserstein
            else:
                dom_idx = None
                dom_energy = 0.0
                dom_wass = 0.0

            link_metrics.append(
                LinkMetrics(
                    link_index=link,
                    onset_index=onset,
                    residual_energy=residual_energy,
                    component_shift=component_shift,
                    detection_index=detection_idx,
                    detection_time=detection_time,
                    detection_delay=detection_delay,
                    dominant_motor=dom_idx,
                    dominant_motor_energy=dom_energy,
                    dominant_motor_wasserstein=dom_wass,
                )
            )

            det_candidate = detection_idx if detection_idx is not None else float("inf")
            detection_candidates.append((det_candidate, -residual_energy, link))

        predicted_link = None
        valid_candidates = [cand for cand in detection_candidates if not np.isinf(cand[0])]
        if valid_candidates:
            predicted_link = min(valid_candidates)[2]
        else:
            energies = [(lm.residual_energy, lm.link_index) for lm in link_metrics]
            if energies:
                predicted_link = max(energies)[1]

        predicted_motor = None
        if predicted_link is not None:
            for lm in link_metrics:
                if lm.link_index == predicted_link:
                    predicted_motor = lm.dominant_motor
                    break

        for lm in link_metrics:
            if lm.onset_index is not None and (true_onset is None or lm.onset_index < true_onset):
                true_onset = lm.onset_index
                true_link = lm.link_index

        true_motor = None
        if true_link is not None and true_onset is not None:
            l0 = true_link * MOTORS_PER_LINK
            l1 = l0 + MOTORS_PER_LINK
            faulty = np.where(labels[true_onset, l0:l1] < 0.5)[0]
            if faulty.size > 0:
                true_motor = int(faulty[0])

        return SequenceReport(
            sample_index=sample_idx,
            predicted_link=predicted_link,
            predicted_motor=predicted_motor,
            true_link=true_link,
            true_motor=true_motor,
            link_metrics=link_metrics,
        )

    def _evaluate_motors(
        self,
        motor_dirs: np.ndarray,
        onset: Optional[int],
        detection_idx: Optional[int],
        dt: float,
    ) -> List[MotorMetrics]:
        T, M, _ = motor_dirs.shape
        if detection_idx is None or detection_idx <= 0:
            return []
        split = onset if onset is not None else T
        detect = detection_idx
        healthy_end = min(split, detect)
        healthy_span = motor_dirs[:healthy_end]
        start_idx = max(split, detect - self.motor_window + 1)
        start_idx = max(0, start_idx)
        end_idx = min(detect + 1 + self.motor_forward_window, T)
        fault_span = motor_dirs[start_idx:end_idx]
        metrics: List[MotorMetrics] = []
        if healthy_span.shape[0] == 0 or fault_span.shape[0] == 0:
            return metrics

        for m in range(M):
            healthy_dirs = healthy_span[:, m, :]
            fault_dirs = fault_span[:, m, :]
            baseline = _unitize(np.mean(healthy_dirs, axis=0, keepdims=True))[0]
            healthy_dists = spherical_distance(healthy_dirs, baseline[None, :])
            fault_dists = spherical_distance(fault_dirs, baseline[None, :])
            geo_energy = geodesic_deviation_energy(fault_dirs, baseline, dt)
            wass = wasserstein_1d(healthy_dists, fault_dists)
            metrics.append(MotorMetrics(motor_index=m, geodesic_energy=geo_energy, wasserstein=wass))
        return metrics

    def _detect_from_energy(self, energy: np.ndarray) -> Optional[int]:
        if energy.size == 0:
            return None
        baseline_end = min(self.baseline_window, energy.size)
        baseline = energy[:baseline_end]
        mean = float(baseline.mean())
        std = float(baseline.std())
        threshold = mean + self.std_scale * max(std, 1e-6)
        peak = float(baseline.max())
        threshold = max(threshold, peak * self.amplitude_scale)
        search = energy[baseline_end:]
        indices = np.where(search >= threshold)[0]
        if indices.size == 0:
            return None
        return int(indices[0] + baseline_end)

    def process_shard(self, shard_path: Path, max_sequences: Optional[int] = None) -> List[SequenceReport]:
        with np.load(shard_path) as data:
            desired_cum = data["desired_link_cum"]  # (S, T, L, 4, 4)
            actual_cum = data["actual_link_cum"]
            labels = data["label"]  # (S, T, L*8)
            dt = float(data["dt"])

        S, T, L, _, _ = desired_cum.shape
        layout = self._ensure_layout(L)
        omega, pos_err = compute_se3_residual(desired_cum, actual_cum)
        rotations = actual_cum[..., :3, :3]
        motor_dirs = compute_motor_directions(rotations, layout)  # (S,T,L,M,3)

        reports: List[SequenceReport] = []
        seq_range = range(S if max_sequences is None else min(S, max_sequences))
        for s in seq_range:
            report = self.evaluate_sequence(
                s,
                omega[s],
                pos_err[s],
                labels[s],
                motor_dirs[s],
                dt,
            )
            reports.append(report)
        return reports


def discover_shards(root: str) -> List[Path]:
    pattern = str(Path(root) / "fault_dataset_shard_*.npz")
    return [Path(p) for p in sorted(glob.glob(pattern))]


def print_report(shard: Path, reports: Sequence[SequenceReport], top_k: int = 3) -> None:
    print(f"\n[shard] {shard.name} | sequences={len(reports)}")
    link_correct = 0
    motor_correct = 0
    motor_total = 0
    for report in reports:
        if report.true_link == report.predicted_link:
            link_correct += 1
        if report.true_motor is not None:
            motor_total += 1
            if report.predicted_link == report.true_link and report.predicted_motor == report.true_motor:
                motor_correct += 1

    acc = link_correct / max(len(reports), 1)
    motor_acc = motor_correct / motor_total if motor_total > 0 else 0.0
    print(f"  accuracy: link={acc*100:.2f}% ({link_correct}/{len(reports)}) motor={motor_acc*100:.2f}% ({motor_correct}/{motor_total})")

    for report in reports[:top_k]:
        print(
            f"  • sample {report.sample_index:04d} → pred link {report.predicted_link} "
            f"(true {report.true_link}) pred motor {report.predicted_motor} (true {report.true_motor})"
        )
        for lm in report.link_metrics:
            flag = "*" if lm.link_index == report.predicted_link else " "
            det_str = "None" if lm.detection_index is None else f"{lm.detection_index} ({lm.detection_time:.3f}s)"
            delay_str = "n/a" if lm.detection_delay is None else f"{lm.detection_delay:.3f}s"
            print(
                f"    {flag} link {lm.link_index}: onset={lm.onset_index} det={det_str} delay={delay_str} "
                f"energy={lm.residual_energy:.4f} ΔH0={lm.component_shift:+d} "
                f"motor={lm.dominant_motor} geoE={lm.dominant_motor_energy:.4f} "
                f"W2={lm.dominant_motor_wasserstein:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LASDRA topological fault diagnostics on shards.")
    parser.add_argument("--data-root", type=str, default="data_storage/link_3", help="Directory containing shards.")
    parser.add_argument("--max-shards", type=int, default=1, help="Maximum number of shards to process.")
    parser.add_argument("--max-seqs", type=int, default=5, help="Maximum sequences per shard (None for all).")
    parser.add_argument("--component-threshold-deg", type=float, default=15.0, help="Adjacency threshold on S^2.")
    parser.add_argument("--baseline-window", type=int, default=100, help="Frames used to estimate healthy energy stats.")
    parser.add_argument("--std-scale", type=float, default=4.0, help="Std-dev multiplier for detection threshold.")
    parser.add_argument("--motor-window", type=int, default=30, help="Frames before detection used for motor ID.")
    parser.add_argument("--motor-forward-window", type=int, default=20, help="Frames after detection used for motor ID.")
    parser.add_argument("--amplitude-scale", type=float, default=12.0, help="Multiplier on healthy peak energy for detection.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of sequences to print per shard.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards = discover_shards(args.data_root)
    if not shards:
        raise FileNotFoundError(f"No shards found in {args.data_root}")
    shards = shards[: args.max_shards]
    pipeline = TopologicalFDIPipeline(
        component_threshold_deg=args.component_threshold_deg,
        baseline_window=args.baseline_window,
        std_scale=args.std_scale,
        motor_window=args.motor_window,
        motor_forward_window=args.motor_forward_window,
        amplitude_scale=args.amplitude_scale,
    )

    for shard in shards:
        reports = pipeline.process_shard(shard, max_sequences=None if args.max_seqs < 0 else args.max_seqs)
        print_report(shard, reports, top_k=args.top_k)


if __name__ == "__main__":
    main()

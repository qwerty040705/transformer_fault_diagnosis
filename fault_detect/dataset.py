# fault_detect/dataset.py
from __future__ import annotations
import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional
from .features import build_link_features

class LASDRAFaultDataset(Dataset):
    """
    Map-style dataset.
    아이템 단위: (샘플 S, 링크 l) 1개를 선택하고, 시간 윈도우를 샘플링해서 반환.
    반환:
      x: (W, F)
      y: (W, 8)  # 링크의 8개 모터에 대한 고장 타깃 (fault=1, normal=0)
      onset_mask: (W, 8)  # 온셋 가중 손실용 마스크(0/1)
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 1234,
        win_size: int = 256,
        normalize: bool = True,
        stats_samples: int = 200,  # 통계 추정에 사용할 샘플 수(최대)
    ):
        super().__init__()
        assert split in ("train", "val")
        self.data_dir = data_dir
        self.win_size = win_size
        self.normalize = normalize

        rng = random.Random(seed)
        paths = sorted(glob.glob(os.path.join(data_dir, "fault_dataset_shard_*.npz")))
        if not paths:
            raise FileNotFoundError(f"No shards in {data_dir}")
        # 간단히 파일 기준으로 split
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_ratio))
        self.val_paths = paths[:n_val]
        self.train_paths = paths[n_val:] if len(paths) - n_val > 0 else paths
        self.paths = self.train_paths if split == "train" else self.val_paths

        # 인덱스 빌드: (path_idx, sample_idx, link_idx)
        self.index: List[Tuple[int, int, int]] = []
        self._file_cache: Dict[int, Dict[str, Any]] = {}

        for pi, p in enumerate(self.paths):
            with np.load(p) as z:
                S = z["desired_link_rel"].shape[0]
                L = z["desired_link_rel"].shape[2]
            for s in range(S):
                for l in range(L):
                    self.index.append((pi, s, l))

        # normalize 통계
        self._mean = None
        self._std = None
        if self.normalize:
            self._mean, self._std = self._estimate_stats(stats_samples)

    def __len__(self):
        return len(self.index)

    def _load_file(self, pi: int) -> Dict[str, Any]:
        if pi in self._file_cache:
            return self._file_cache[pi]
        path = self.paths[pi]
        z = np.load(path)
        # lazy: 그대로 캐시에 보관 (메모리 충분한 경우)
        self._file_cache[pi] = {k: z[k] for k in z.files}
        return self._file_cache[pi]

    def _estimate_stats(self, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        feats_list = []
        cnt = 0
        for pi, p in enumerate(self.paths):
            f = self._load_file(pi)
            S = f["desired_link_rel"].shape[0]
            L = f["desired_link_rel"].shape[2]
            dt = float(f["dt"])
            for s in range(min(S, max_samples)):
                # 한 링크만 샘플
                l = 0
                feats = build_link_features(
                    f["desired_link_rel"][s], f["actual_link_rel"][s],
                    f["desired_link_cum"][s], f["actual_link_cum"][s],
                    f["desired_ee"][s], f["actual_ee"][s],
                    dt
                )  # (T,L,F)
                feats_list.append(feats[:, l, :])
                cnt += 1
                if cnt >= max_samples:
                    break
            if cnt >= max_samples:
                break
        X = np.concatenate(feats_list, axis=0)  # (sum_T, F)
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-6
        return mean.astype(np.float32), std.astype(np.float32)

    def _slice_labels(self, label: np.ndarray, link_idx: int) -> np.ndarray:
        """
        label: (T, 8*L) with 1=normal, 0=fault
        return fault target: (T, 8) with 1=fault, 0=normal
        """
        L = label.shape[1] // 8
        j0 = link_idx * 8
        y = label[:, j0:j0+8]
        return (1 - y).astype(np.float32)  # fault target

    def _build_onset_mask(self, onset_idx: np.ndarray, link_idx: int, T: int, radius: int = 3) -> np.ndarray:
        """
        onset_idx: (8*L,) with -1 meaning no fault
        return: (T,8) mask where positions near onset are 1
        """
        L = onset_idx.shape[0] // 8
        j0 = link_idx * 8
        on = onset_idx[j0:j0+8]
        mask = np.zeros((T, 8), dtype=np.float32)
        for m in range(8):
            o = int(on[m])
            if o >= 0 and o < T:
                lo = max(0, o - radius)
                hi = min(T, o + radius + 1)
                mask[lo:hi, m] = 1.0
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pi, s, l = self.index[idx]
        f = self._load_file(pi)

        dt = float(f["dt"])
        T = f["desired_link_rel"].shape[1]  # careful: (S, T, L, 4, 4) -> axis 1 is T
        # numpy 저장축: (S, T, L, 4,4)
        d_lr = f["desired_link_rel"][s]  # (T, L, 4,4)
        a_lr = f["actual_link_rel"][s]
        d_lc = f["desired_link_cum"][s]
        a_lc = f["actual_link_cum"][s]
        d_ee = f["desired_ee"][s]       # (T, 4,4)
        a_ee = f["actual_ee"][s]
        label = f["label"][s]           # (T, 8L)
        onset = f["onset_idx"][s]       # (8L,)

        feats = build_link_features(d_lr, a_lr, d_lc, a_lc, d_ee, a_ee, dt)  # (T,L,F)
        x_full = feats[:, l, :]  # (T,F)
        y_full = self._slice_labels(label, l)  # (T,8)
        onset_mask_full = self._build_onset_mask(onset, l, T)

        W = min(self.win_size, x_full.shape[0])
        if W < x_full.shape[0]:
            # 랜덤 윈도우 샘플링
            start = random.randint(0, x_full.shape[0] - W)
        else:
            start = 0

        x = x_full[start:start+W]
        y = y_full[start:start+W]
        onset_mask = onset_mask_full[start:start+W]

        if self.normalize and (self._mean is not None):
            x = (x - self._mean) / self._std

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        onset_mask = torch.from_numpy(onset_mask).float()
        return {"x": x, "y": y, "onset_mask": onset_mask}

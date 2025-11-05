#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LASDRA 모터 고장 탐지 - 단일 파일 학습 스크립트
- 데이터: data_storage/link_{L}/fault_dataset_shard_*.npz
- 특징: SE(3) 잔차(자세 rotvec 3 + 위치 3) × (rel, cum, ee) + 시간미분 = 36차원/링크
- 모델: TCN(팽창 1D-Conv) + Transformer Encoder (경량)
- 손실: BCE(pos_weight) + 온셋 가중 + 시계열 평활(Total Variation)
- 성능/안정: CUDA 학습, 특징은 CPU에서 LRU 캐시, CPU 스레드 1, workers 기본 0
"""

from __future__ import annotations
import os, sys, argparse, math, glob, random, time
from collections import OrderedDict
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 시스템/스레드 최적화: 과도한 BLAS 스레드로 인한 랙 방지
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import amp
from typing import Tuple, Dict, Any, List

torch.set_num_threads(1)  # CPU 병렬 과점 방지(Ampere/GPU에서 주로 CUDA가 bottleneck)
torch.set_float32_matmul_precision('high')  # TF32 허용 (Ampere+)

# ──────────────────────────────────────────────────────────────────────────────
# 수학 헬퍼: SO(3)/SE(3) 잔차(전부 NumPy, CPU 전용 → DataLoader에서 CUDA 초기화 방지)
# ──────────────────────────────────────────────────────────────────────────────
def _vee_np(S: np.ndarray) -> np.ndarray:
    # S: (...,3,3) skew-symmetric
    # return (...,3)
    x = np.stack([
        S[..., 2, 1] - S[..., 1, 2],
        S[..., 0, 2] - S[..., 2, 0],
        S[..., 1, 0] - S[..., 0, 1]
    ], axis=-1)
    return x

def so3_log_np(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    R: (...,3,3)
    return: (...,3) (axis * angle)
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(cos_theta)  # (...,)

    S = 0.5 * (R - np.swapaxes(R, -1, -2))
    v = _vee_np(S)  # (...,3)

    small = (theta < 1e-3)
    sin_theta = np.sin(theta)
    scale = (theta / (sin_theta + eps))[..., None]  # (...,1)
    v_general = v * scale

    out = v_general.copy()
    if np.any(small):
        out[small] = v[small]  # 작은 각에서는 1차 근사
    return out

def se3_err_np(T_act: np.ndarray, T_des: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    T_act, T_des: (...,4,4)
    Returns:
      e_r: (...,3)  (rotvec of R_des^T R_act)
      e_p: (...,3)  (p_act - p_des)
    """
    R_act = T_act[..., :3, :3]
    R_des = T_des[..., :3, :3]
    p_act = T_act[..., :3, 3]
    p_des = T_des[..., :3, 3]
    R_err = np.matmul(np.swapaxes(R_des, -1, -2), R_act)  # R_des^T R_act
    e_r = so3_log_np(R_err)
    e_p = p_act - p_des
    return e_r, e_p

def diff_time_np(x: np.ndarray, dt: float) -> np.ndarray:
    dx = np.diff(x, axis=0) / dt
    first = np.zeros_like(dx[:1])
    return np.concatenate([first, dx], axis=0)

def build_link_features_np(
    desired_link_rel: np.ndarray, actual_link_rel: np.ndarray,
    desired_link_cum: np.ndarray, actual_link_cum: np.ndarray,
    desired_ee: np.ndarray, actual_ee: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    *_link_rel/cum: (T, L, 4, 4), *_ee: (T, 4, 4)
    return feats: (T, L, 36)
    """
    T, L = desired_link_rel.shape[:2]

    # EE를 L로 타일
    Td_ee = np.broadcast_to(desired_ee[:, None, :, :], (T, L, 4, 4))
    Ta_ee = np.broadcast_to(actual_ee[:, None, :, :], (T, L, 4, 4))

    # (T,L,4,4) -> (T*L,4,4)
    def flat(x): return x.reshape(-1, 4, 4)
    e_rr, e_rp = se3_err_np(flat(actual_link_rel), flat(desired_link_rel))
    e_cr, e_cp = se3_err_np(flat(actual_link_cum), flat(desired_link_cum))
    e_er, e_ep = se3_err_np(flat(Ta_ee), flat(Td_ee))

    # (T*L,3) -> (T,L,3)
    def unflat(x): return x.reshape(T, L, 3)
    e_rr, e_rp = unflat(e_rr), unflat(e_rp)
    e_cr, e_cp = unflat(e_cr), unflat(e_cp)
    e_er, e_ep = unflat(e_er), unflat(e_ep)

    base = np.concatenate([e_rr, e_rp, e_cr, e_cp, e_er, e_ep], axis=-1)  # (T,L,18)

    d_rr = diff_time_np(e_rr, dt)
    d_rp = diff_time_np(e_rp, dt)
    d_cr = diff_time_np(e_cr, dt)
    d_cp = diff_time_np(e_cp, dt)
    d_er = diff_time_np(e_er, dt)
    d_ep = diff_time_np(e_ep, dt)

    d_feats = np.concatenate([d_rr, d_rp, d_cr, d_cp, d_er, d_ep], axis=-1)  # (T,L,18)
    feats = np.concatenate([base, d_feats], axis=-1).astype(np.float32)      # (T,L,36)
    return feats

# ──────────────────────────────────────────────────────────────────────────────
# 데이터셋 (단일 파일로 통합: 특징 LRU 캐시, 파일 캐시)
# ──────────────────────────────────────────────────────────────────────────────
class LRU:
    def __init__(self, capacity: int = 64):
        self.cap = capacity
        self.od: OrderedDict = OrderedDict()
    def get(self, k):
        v = self.od.get(k)
        if v is not None:
            self.od.move_to_end(k)
        return v
    def put(self, k, v):
        self.od[k] = v
        self.od.move_to_end(k)
        while len(self.od) > self.cap:
            self.od.popitem(last=False)

class LASDRAFaultDataset(Dataset):
    """
    아이템: (파일 pi, 샘플 s, 링크 l)에서 윈도우 추출
    반환:
      x: (W, F=36), y: (W, 8), onset_mask: (W,8)
    특징은 (pi,s) 단위로 캐시되어 중복계산 방지
    """
    def __init__(self, data_dir: str, split: str, win_size: int,
                 val_ratio: float = 0.1, seed: int = 1234, cache_size: int = 128,
                 normalize: bool = True):
        super().__init__()
        assert split in ("train", "val")
        self.data_dir = data_dir
        self.win_size = win_size
        self.normalize = normalize

        rng = random.Random(seed)
        self.paths = sorted(glob.glob(os.path.join(data_dir, "fault_dataset_shard_*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"No shards in {data_dir}")
        rng.shuffle(self.paths)
        n_val = max(1, int(len(self.paths) * val_ratio))
        self.val_paths = self.paths[:n_val]
        self.train_paths = self.paths[n_val:] if len(self.paths) > n_val else self.paths
        self.paths = self.train_paths if split == "train" else self.val_paths

        # 파일 단위 Shape 수집 및 인덱스 생성
        self.file_meta: Dict[int, Dict[str, Any]] = {}
        self.index: List[Tuple[int, int, int]] = []  # (pi, s, l)
        for pi, p in enumerate(self.paths):
            with np.load(p) as z:
                S = z["desired_link_rel"].shape[0]
                T = z["desired_link_rel"].shape[1]
                L = z["desired_link_rel"].shape[2]
            self.file_meta[pi] = {"S": S, "T": T, "L": L}
            for s in range(S):
                for l in range(L):
                    self.index.append((pi, s, l))

        # LRU 캐시: (pi,s) → {"feats":(T,L,F), "y_full":(T,8L), "onset":(8L), "dt":float}
        self.sample_cache = LRU(capacity=cache_size)

        # 정규화 통계 (가볍게 일부 샘플로 추정)
        self.mean = None
        self.std = None
        if self.normalize:
            self.mean, self.std = self._estimate_stats(max_samples=min(200, len(self.index)))

    def _estimate_stats(self, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        feats_list = []
        picked = 0
        for pi, p in enumerate(self.paths):
            with np.load(p) as z:
                S = z["desired_link_rel"].shape[0]
                L = z["desired_link_rel"].shape[2]
                dt = float(z["dt"])
                for s in range(S):
                    if picked >= max_samples: break
                    feats = self._get_or_build(pi, s, z)["feats"]  # (T,L,F)
                    # 한 링크만 써도 통계엔 충분
                    feats_list.append(feats[:, 0, :])
                    picked += 1
            if picked >= max_samples:
                break
        X = np.concatenate(feats_list, axis=0)
        mean = X.mean(axis=0).astype(np.float32)
        std = (X.std(axis=0) + 1e-6).astype(np.float32)
        return mean, std

    def _get_or_build(self, pi: int, s: int, z_loaded: Dict[str, Any] | None = None) -> Dict[str, Any]:
        key = (pi, s)
        cached = self.sample_cache.get(key)
        if cached is not None:
            return cached

        # 파일 열기(이미 열려있으면 재사용)
        if z_loaded is None:
            z = np.load(self.paths[pi])
        else:
            z = z_loaded

        d_lr = z["desired_link_rel"][s]  # (T,L,4,4)
        a_lr = z["actual_link_rel"][s]
        d_lc = z["desired_link_cum"][s]
        a_lc = z["actual_link_cum"][s]
        d_ee = z["desired_ee"][s]       # (T,4,4)
        a_ee = z["actual_ee"][s]
        dt   = float(z["dt"])
        label = z["label"][s]           # (T, 8L) 1=정상, 0=고장
        onset = z["onset_idx"][s]       # (8L,)

        feats = build_link_features_np(d_lr, a_lr, d_lc, a_lc, d_ee, a_ee, dt)  # (T,L,F)
        entry = {"feats": feats, "y_full": (1 - label).astype(np.float32), "onset": onset.astype(np.int32), "dt": dt}
        self.sample_cache.put(key, entry)
        return entry

    @staticmethod
    def _slice_labels(y_full: np.ndarray, link_idx: int) -> np.ndarray:
        # y_full: (T, 8L) fault target(1=고장)
        L = y_full.shape[1] // 8
        j0 = link_idx * 8
        return y_full[:, j0:j0+8]

    @staticmethod
    def _onset_mask(onset: np.ndarray, link_idx: int, T: int, radius: int = 3) -> np.ndarray:
        L = onset.shape[0] // 8
        j0 = link_idx * 8
        on = onset[j0:j0+8]
        mask = np.zeros((T, 8), dtype=np.float32)
        for m in range(8):
            o = int(on[m])
            if 0 <= o < T:
                lo = max(0, o - radius)
                hi = min(T, o + radius + 1)
                mask[lo:hi, m] = 1.0
        return mask

    def __len__(self): return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pi, s, l = self.index[idx]
        with np.load(self.paths[pi]) as z:
            entry = self._get_or_build(pi, s, z_loaded=z)
        feats = entry["feats"]           # (T,L,F)
        y_full = entry["y_full"]         # (T,8L)
        onset = entry["onset"]           # (8L,)
        dt = entry["dt"]
        T, L, F = feats.shape

        x_full = feats[:, l, :]          # (T,F)
        y_link = self._slice_labels(y_full, l)     # (T,8)
        om_full = self._onset_mask(onset, l, T)    # (T,8)

        W = min(self.win_size, T)
        if W < T:
            # train/val 동일하게 윈도우 샘플 (간단/빠름)
            start = random.randint(0, T - W)
        else:
            start = 0
        x = x_full[start:start+W]
        y = y_link[start:start+W]
        om = om_full[start:start+W]

        if self.normalize and (self.mean is not None):
            x = (x - self.mean) / self.std

        # torch tensor
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        om = torch.from_numpy(om).float()
        return {"x": x, "y": y, "onset_mask": om}

# ──────────────────────────────────────────────────────────────────────────────
# 불균형 가중 추정 (DataLoader 없이 빠르게)
# ──────────────────────────────────────────────────────────────────────────────
def estimate_pos_weight_from_files(paths: List[str], use_paths: List[str]) -> float:
    pos = 0.0  # fault=1
    neg = 0.0  # normal=1
    for p in use_paths:
        with np.load(p) as z:
            lab = z["label"]  # (S,T,8L) 1=정상, 0=고장
            pos += (1.0 - lab).sum()
            neg += lab.sum()
    pos = max(float(pos), 1.0)
    neg = max(float(neg), 1.0)
    return neg / pos

# ──────────────────────────────────────────────────────────────────────────────
# 모델 (TCN + Transformer)
# ──────────────────────────────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, dropout=0.1):
        super().__init__()
        k = 3
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):  # (B,C,T)
        y = self.drop(self.act(self.bn(self.conv(x))))
        return y + self.res(x)

class LinkTemporalModel(nn.Module):
    def __init__(self, in_dim=36, hidden=160, nheads=8, nlayers=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.tcn = nn.Sequential(
            TCNBlock(hidden, hidden, dilation=1, dropout=dropout),
            TCNBlock(hidden, hidden, dilation=2, dropout=dropout),
            TCNBlock(hidden, hidden, dilation=4, dropout=dropout),
            TCNBlock(hidden, hidden, dilation=8, dropout=dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nheads, batch_first=True,
            dim_feedforward=hidden*4, dropout=dropout, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 8)
        )
    def forward(self, x):  # x: (B,T,F)
        h = self.in_proj(x)       # (B,T,H)
        h = self.tcn(h.transpose(1,2)).transpose(1,2)  # (B,T,H)
        h = self.encoder(h)       # (B,T,H)
        return self.head(h)       # (B,T,8)

# ──────────────────────────────────────────────────────────────────────────────
# 손실/지표
# ──────────────────────────────────────────────────────────────────────────────
def onset_weighted_bce(logits: torch.Tensor, target: torch.Tensor, onset_mask: torch.Tensor, weight: float = 3.0):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    w = 1.0 + (weight - 1.0) * onset_mask
    return (bce * w).mean()

def total_variation_loss(probs: torch.Tensor, lam: float = 0.05):
    diff = probs[:, 1:, :] - probs[:, :-1, :]
    return lam * diff.abs().mean()

def compute_metrics_np(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.uint8)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    # F1
    tp = np.sum((yt==1) & (yp==1))
    tn = np.sum((yt==0) & (yp==0))
    fp = np.sum((yt==0) & (yp==1))
    fn = np.sum((yt==1) & (yp==0))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec / (prec + rec + 1e-9)
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    bacc = 0.5 * (tpr + tnr)
    return {"f1": float(f1), "bacc": float(bacc), "auroc": float("nan")}

# ──────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────────────────────────────────────
def train_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} cuda={torch.cuda.is_available()} torch={torch.__version__}")

    # Dataset / Dataloader
    ds_train = LASDRAFaultDataset(args.data_dir, split="train", win_size=args.win_size,
                                  val_ratio=args.val_ratio, cache_size=args.feature_cache, normalize=True)
    ds_val   = LASDRAFaultDataset(args.data_dir, split="val",   win_size=args.win_size,
                                  val_ratio=args.val_ratio, cache_size=args.feature_cache, normalize=True)

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=(args.workers>0),
        prefetch_factor=(2 if args.workers>0 else None)
    )
    dl_val   = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=False, persistent_workers=(args.workers>0),
        prefetch_factor=(2 if args.workers>0 else None)
    )

    # pos_weight (파일 순회)
    tr_paths = ds_train.paths
    pos_weight = estimate_pos_weight_from_files(ds_train.paths + ds_val.paths, tr_paths)
    print(f"[INFO] pos_weight ≈ {pos_weight:.2f}")

    model = LinkTemporalModel(in_dim=36, hidden=args.hidden, nheads=8, nlayers=2, dropout=args.dropout).to(device)

    # (선택) torch.compile — 기본 비활성. 켜면 첫 epoch 시작이 느려질 수 있음.
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile disabled: {e}")

    scaler = amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_bacc = -1.0
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        for i, batch in enumerate(dl_train, start=1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            om = batch["onset_mask"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=(device.type=="cuda")):
                logits = model(x)                     # (B,W,8)
                loss_b = bce_loss(logits, y)
                loss_o = onset_weighted_bce(logits, y, om, weight=args.onset_weight)
                probs = torch.sigmoid(logits)
                loss_tv = total_variation_loss(probs, lam=args.tv_lambda)
                loss = loss_b + loss_o + loss_tv

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            running += float(loss.item())
            if (i % args.log_every) == 0:
                speed = (i * args.batch_size) / max(1e-9, (time.time()-t0))
                print(f"[Train][Ep {epoch:02d}] step {i:04d}/{len(dl_train):04d} "
                      f"loss={running/i:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"speed~{speed:.1f} samp/s")

        sched.step()

        # ── Validation
        model.eval()
        with torch.no_grad():
            all_true, all_prob = [], []
            for batch in dl_val:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                with amp.autocast('cuda', enabled=(device.type=="cuda")):
                    prob = torch.sigmoid(model(x))
                all_true.append(y.cpu().numpy())
                all_prob.append(prob.cpu().numpy())
            Y = np.concatenate(all_true, axis=0).reshape(-1, 8)
            P = np.concatenate(all_prob, axis=0).reshape(-1, 8)
            metrics = compute_metrics_np(Y, P, threshold=args.eval_threshold)
            bacc = metrics["bacc"]
            print(f"[Val][Ep {epoch:02d}] bacc={bacc:.4f} f1={metrics['f1']:.4f} auroc={metrics['auroc']}")

            if bacc > best_bacc:
                best_bacc = bacc
                # state_dict만 저장(compile unwrap 자동 처리)
                torch.save({"model": (model.module.state_dict() if isinstance(model, nn.DataParallel) else
                                      model.state_dict()),
                            "args": vars(args)}, best_path)
                print(f"  ✅ Saved best to {best_path}")

    print(f"Done. Best balanced accuracy={best_bacc:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="data_storage/link_{L}")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--win_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0, help="권장 0~2 (WSL/데스크탑 랙 방지)")
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--onset_weight", type=float, default=3.0)
    ap.add_argument("--tv_lambda", type=float, default=0.05)
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--feature_cache", type=int, default=128, help="(pi,s) LRU 캐시 크기")
    ap.add_argument("--compile", action="store_true", help="torch.compile 활성화(초기 오버헤드 有)")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()
    train_main(args)

if __name__ == "__main__":
    main()

"""
python3 train_lasdra_fault_single.py \
  --data_dir data_storage/link_3 \
  --epochs 25 \
  --batch_size 24 \
  --win_size 256 \
  --workers 0 \
  --log_every 1
"""
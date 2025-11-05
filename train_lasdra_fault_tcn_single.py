#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LASDRA 모터 고장 탐지 - TCN(딥러닝) 단일 파일 학습 스크립트 (Transformer 미사용, 더 빠름)
입력: data_storage/link_{L}/fault_dataset_shard_*.npz
특징: SE(3) 잔차(회전벡터 3 + 위치 3) × (rel, cum, ee) + 시간미분 = 36차원/링크
모델: Dilated Depthwise-Separable TCN + Squeeze-Excitation (채널 주의) + Residual
손실: 가중 BCE(불균형) + 온셋 가중 + Total Variation(시간 평활)
성능/안정: CUDA 학습, 특징은 CPU에서 LRU 캐시(중복계산 제거), BLAS 스레드 1, workers 기본 0
"""

from __future__ import annotations
import os, sys, argparse, glob, random, time
from collections import OrderedDict
from typing import Tuple, Dict, Any, List
import numpy as np

# ── 시스템 스레드/메모리 최적화(랙 방지) ─────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import amp

torch.set_num_threads(1)                        # CPU 병렬 과점 방지
torch.set_float32_matmul_precision('high')      # TF32 가속(Ampere+)

# ── SE(3) 잔차: NumPy(=CPU)로만 계산 → DataLoader에서 CUDA 초기화 금지 ────────────
def _vee_np(S: np.ndarray) -> np.ndarray:
    return np.stack([S[...,2,1]-S[...,1,2], S[...,0,2]-S[...,2,0], S[...,1,0]-S[...,0,1]], axis=-1)

def so3_log_np(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    tr = R[...,0,0] + R[...,1,1] + R[...,2,2]
    cos_th = np.clip((tr - 1.0)*0.5, -1.0+1e-7, 1.0-1e-7)
    th = np.arccos(cos_th)
    S = 0.5*(R - np.swapaxes(R, -1, -2))
    v = _vee_np(S)
    small = (th < 1e-3)
    st = np.sin(th)
    scale = (th/(st+eps))[...,None]
    vg = v*scale
    out = vg.copy()
    if np.any(small): out[small] = v[small]
    return out

def se3_err_np(Ta: np.ndarray, Td: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Ra, Rd = Ta[..., :3,:3], Td[..., :3,:3]
    pa, pd = Ta[..., :3, 3], Td[..., :3, 3]
    Rerr = np.matmul(np.swapaxes(Rd, -1, -2), Ra)
    er = so3_log_np(Rerr)
    ep = pa - pd
    return er, ep

def diff_time_np(x: np.ndarray, dt: float) -> np.ndarray:
    dx = np.diff(x, axis=0) / dt
    first = np.zeros_like(dx[:1])
    return np.concatenate([first, dx], axis=0)

def build_link_features_np(d_rel, a_rel, d_cum, a_cum, d_ee, a_ee, dt: float) -> np.ndarray:
    """
    *_rel/cum: (T,L,4,4), *_ee: (T,4,4) -> feats: (T,L,36)
    """
    T, L = d_rel.shape[:2]
    Td_ee = np.broadcast_to(d_ee[:,None,:,:], (T,L,4,4))
    Ta_ee = np.broadcast_to(a_ee[:,None,:,:], (T,L,4,4))

    def flat(x):  return x.reshape(-1,4,4)
    def unflat(x):return x.reshape(T,L,3)

    e_rr, e_rp = se3_err_np(flat(a_rel), flat(d_rel)); e_rr, e_rp = unflat(e_rr), unflat(e_rp)
    e_cr, e_cp = se3_err_np(flat(a_cum), flat(d_cum)); e_cr, e_cp = unflat(e_cr), unflat(e_cp)
    e_er, e_ep = se3_err_np(flat(Ta_ee), flat(Td_ee)); e_er, e_ep = unflat(e_er), unflat(e_ep)

    base = np.concatenate([e_rr,e_rp,e_cr,e_cp,e_er,e_ep], axis=-1)  # (T,L,18)
    d_feats = np.concatenate([
        diff_time_np(e_rr, dt), diff_time_np(e_rp, dt),
        diff_time_np(e_cr, dt), diff_time_np(e_cp, dt),
        diff_time_np(e_er, dt), diff_time_np(e_ep, dt)
    ], axis=-1)                                                     # (T,L,18)
    feats = np.concatenate([base, d_feats], axis=-1).astype(np.float32)  # (T,L,36)
    return feats

# ── 간단 LRU: (pi,s) 단위 특징 캐시 ─────────────────────────────────────────────
class LRU:
    def __init__(self, capacity: int = 128):
        self.cap = capacity
        self.od: OrderedDict = OrderedDict()
    def get(self, k): 
        v = self.od.get(k)
        if v is not None: self.od.move_to_end(k)
        return v
    def put(self, k, v):
        self.od[k] = v; self.od.move_to_end(k)
        while len(self.od) > self.cap: self.od.popitem(last=False)

# ── 데이터셋 ───────────────────────────────────────────────────────────────────
class LASDRAFaultDataset(Dataset):
    """
    아이템: (파일 pi, 샘플 s, 링크 l)에서 길이 W 윈도우 슬라이스
    반환: x:(W,36) y:(W,8) onset_mask:(W,8)
    """
    def __init__(self, data_dir: str, split: str, win_size: int,
                 val_ratio: float = 0.1, seed: int = 1234, cache_size: int = 128,
                 normalize: bool = True, mean_std: Tuple[np.ndarray,np.ndarray] | None = None,
                 mmap: bool = False):
        assert split in ("train","val")
        self.win_size = win_size
        self.normalize = normalize
        self.mmap = mmap

        rng = random.Random(seed)
        paths = sorted(glob.glob(os.path.join(data_dir, "fault_dataset_shard_*.npz")))
        if not paths: raise FileNotFoundError(f"No shards in {data_dir}")
        rng.shuffle(paths)
        n_val = max(1, int(len(paths)*val_ratio))
        self.val_paths  = paths[:n_val]
        self.train_paths= paths[n_val:] if len(paths)>n_val else paths
        self.paths = self.train_paths if split=="train" else self.val_paths

        # 메타/인덱스
        self.file_meta: Dict[int,Dict[str,Any]] = {}
        self.index: List[Tuple[int,int,int]] = []
        for pi,p in enumerate(self.paths):
            with np.load(p, mmap_mode='r' if mmap else None) as z:
                S = z["desired_link_rel"].shape[0]
                T = z["desired_link_rel"].shape[1]
                L = z["desired_link_rel"].shape[2]
            self.file_meta[pi] = {"S":S,"T":T,"L":L}
            for s in range(S):
                for l in range(L):
                    self.index.append((pi,s,l))

        self.cache = LRU(cache_size)
        self.mean, self.std = (None, None)
        if mean_std is not None:
            self.mean, self.std = mean_std
        elif self.normalize:
            self.mean, self.std = self._estimate_stats(max_samples=min(200,len(self.index)))

    def _estimate_stats(self, max_samples:int)->Tuple[np.ndarray,np.ndarray]:
        feats_list = []; picked=0
        for pi,p in enumerate(self.paths):
            with np.load(p, mmap_mode='r' if self.mmap else None) as z:
                S = z["desired_link_rel"].shape[0]; L = z["desired_link_rel"].shape[2]
                for s in range(S):
                    if picked>=max_samples: break
                    feats = self._get_or_build(pi,s,z)["feats"]  # (T,L,F)
                    feats_list.append(feats[:,0,:]); picked+=1
            if picked>=max_samples: break
        X = np.concatenate(feats_list, axis=0)
        mean = X.mean(axis=0).astype(np.float32)
        std  = (X.std(axis=0)+1e-6).astype(np.float32)
        return mean, std

    def _get_or_build(self, pi:int, s:int, z_loaded=None)->Dict[str,Any]:
        key=(pi,s); got=self.cache.get(key)
        if got is not None: return got
        z = z_loaded if z_loaded is not None else np.load(self.paths[pi], mmap_mode='r' if self.mmap else None)
        d_lr = z["desired_link_rel"][s]; a_lr = z["actual_link_rel"][s]
        d_lc = z["desired_link_cum"][s]; a_lc = z["actual_link_cum"][s]
        d_ee = z["desired_ee"][s];       a_ee = z["actual_ee"][s]
        dt   = float(z["dt"])
        lab  = z["label"][s]              # (T,8L), 1=정상
        onset= z["onset_idx"][s]          # (8L,)
        feats = build_link_features_np(d_lr,a_lr,d_lc,a_lc,d_ee,a_ee,dt)  # (T,L,36)
        entry={"feats":feats,"y_full":(1-lab).astype(np.float32),"onset":onset.astype(np.int32),"dt":dt}
        self.cache.put(key,entry); return entry

    def __len__(self): return len(self.index)

    @staticmethod
    def _slice_labels(y_full:np.ndarray, link_idx:int)->np.ndarray:
        L = y_full.shape[1]//8; j0=link_idx*8
        return y_full[:, j0:j0+8]

    @staticmethod
    def _onset_mask(onset:np.ndarray, link_idx:int, T:int, radius:int=3)->np.ndarray:
        L = onset.shape[0]//8; j0=link_idx*8; on=onset[j0:j0+8]
        m = np.zeros((T,8),dtype=np.float32)
        for mtr in range(8):
            o=int(on[mtr])
            if 0<=o<T:
                lo=max(0,o-radius); hi=min(T,o+radius+1); m[lo:hi,mtr]=1.0
        return m

    def __getitem__(self, idx:int)->Dict[str,torch.Tensor]:
        pi,s,l = self.index[idx]
        with np.load(self.paths[pi], mmap_mode='r' if self.mmap else None) as z:
            entry=self._get_or_build(pi,s,z)
        feats=entry["feats"]; y_full=entry["y_full"]; onset=entry["onset"]; T=feats.shape[0]
        x_full=feats[:,l,:]; y_link=self._slice_labels(y_full,l); om_full=self._onset_mask(onset,l,T)

        W=min(self.win_size,T)
        start = random.randint(0, T-W) if W<T else 0
        x = x_full[start:start+W]; y=y_link[start:start+W]; om=om_full[start:start+W]

        if self.normalize and (self.mean is not None):
            x = (x - self.mean)/self.std

        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(y).float(),
            "onset_mask": torch.from_numpy(om).float()
        }

# ── 불균형 가중(고장 희소) ─────────────────────────────────────────────────────
def estimate_pos_weight_from_files(paths: List[str]) -> float:
    pos = 0.0; neg = 0.0
    for p in paths:
        with np.load(p, mmap_mode='r') as z:
            lab = z["label"]  # (S,T,8L) 1=정상
            pos += (1.0 - lab).sum()
            neg += lab.sum()
    pos=max(float(pos),1.0); neg=max(float(neg),1.0)
    return neg/pos

# ── 모델: Dilated Depthwise-Separable TCN + SE Attention ──────────────────────
class SEBlock(nn.Module):
    def __init__(self, C: int, r: int = 8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)          # (B,C,T)->(B,C,1)
        self.fc  = nn.Sequential(
            nn.Conv1d(C, max(1,C//r), kernel_size=1),
            nn.GELU(),
            nn.Conv1d(max(1,C//r), C, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):  # (B,C,T)
        w = self.fc(self.avg(x))
        return x * w

class DWSepTCNBlock(nn.Module):
    def __init__(self, C: int, dilation: int, dropout: float=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=C)
        self.pwglu = nn.Conv1d(C, 2*C, kernel_size=1)   # GLU 게이트
        self.dwconv= nn.Conv1d(C, C, kernel_size=3, padding=dilation, dilation=dilation, groups=C)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=C)
        self.se    = SEBlock(C)
        self.pw    = nn.Conv1d(C, C, kernel_size=1)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):  # (B,C,T)
        h = self.norm1(x)
        a,b = self.pwglu(h).chunk(2, dim=1)
        h = a * torch.sigmoid(b)             # GLU
        h = self.dwconv(h)
        h = self.norm2(h)
        h = torch.nn.functional.gelu(h)
        h = self.se(h)
        h = self.pw(h)
        h = self.drop(h)
        return x + h

class TCNModel(nn.Module):
    def __init__(self, in_dim=36, hidden=160, dilations:List[int]|None=None, n_blocks:int=8, dropout:float=0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        if dilations is None:
            dilations = [1,2,4,8,16,32,64,128][:n_blocks]
        self.tcn = nn.ModuleList([DWSepTCNBlock(hidden, d, dropout=dropout) for d in dilations])
        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 8, kernel_size=1)
        )
    def forward(self, x):  # x: (B,T,F)
        h = self.inp(x)            # (B,T,H)
        h = h.transpose(1,2)       # (B,H,T)
        for blk in self.tcn:
            h = blk(h)             # (B,H,T)
        out = self.head(h).transpose(1,2)  # (B,T,8)
        return out

# ── 손실 & 지표 ───────────────────────────────────────────────────────────────
def onset_weighted_bce(logits: torch.Tensor, target: torch.Tensor, onset_mask: torch.Tensor, weight: float = 3.0):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    w = 1.0 + (weight - 1.0) * onset_mask
    return (bce * w).mean()

def total_variation_loss(probs: torch.Tensor, lam: float = 0.05):
    diff = probs[:,1:,:] - probs[:,:-1,:]
    return lam * diff.abs().mean()

def compute_metrics_np(y_true: np.ndarray, y_prob: np.ndarray, thr: float=0.5)->Dict[str,float]:
    y_pred = (y_prob >= thr).astype(np.uint8)
    yt = y_true.reshape(-1); yp = y_pred.reshape(-1)
    tp = np.sum((yt==1)&(yp==1)); tn=np.sum((yt==0)&(yp==0))
    fp = np.sum((yt==0)&(yp==1)); fn=np.sum((yt==1)&(yp==0))
    prec = tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9)
    tpr = tp/(tp+fn+1e-9); tnr=tn/(tn+fp+1e-9); bacc=0.5*(tpr+tnr)
    return {"f1":float(f1), "bacc":float(bacc), "auroc": float("nan")}

# ── 학습 루프 ──────────────────────────────────────────────────────────────────
def train_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} cuda={torch.cuda.is_available()} torch={torch.__version__}")

    # Dataset (train 먼저 만들어 통계를 공유)
    ds_train = LASDRAFaultDataset(args.data_dir, split="train", win_size=args.win_size,
                                  val_ratio=args.val_ratio, cache_size=args.feature_cache,
                                  normalize=True, mean_std=None, mmap=args.mmap)
    mean_std = (ds_train.mean, ds_train.std)
    ds_val   = LASDRAFaultDataset(args.data_dir, split="val",   win_size=args.win_size,
                                  val_ratio=args.val_ratio, cache_size=args.feature_cache,
                                  normalize=True, mean_std=mean_std, mmap=args.mmap)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=(args.workers>0),
                          prefetch_factor=(2 if args.workers>0 else None))
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True, drop_last=False,
                          persistent_workers=(args.workers>0),
                          prefetch_factor=(2 if args.workers>0 else None))

    # 불균형 가중치
    pos_weight = estimate_pos_weight_from_files(ds_train.paths)
    print(f"[INFO] pos_weight ≈ {pos_weight:.2f}")

    model = TCNModel(in_dim=36, hidden=args.hidden, n_blocks=args.tcn_blocks, dropout=args.dropout).to(device)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile disabled: {e}")

    scaler = amp.GradScaler('cuda', enabled=(device.type=="cuda"))
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_bacc = -1.0
    best_path = os.path.join(args.ckpt_dir, "best_tcn.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time(); running=0.0; steps=len(dl_train)

        for i, batch in enumerate(dl_train, start=1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            om= batch["onset_mask"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=(device.type=="cuda")):
                logits = model(x)                     # (B,W,8)
                loss_b = bce_loss(logits, y)
                loss_o = onset_weighted_bce(logits, y, om, weight=args.onset_weight)
                probs  = torch.sigmoid(logits)
                loss_tv= total_variation_loss(probs, lam=args.tv_lambda)
                loss   = loss_b + loss_o + loss_tv

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()

            running += float(loss.item())
            if (i % args.log_every)==0:
                elapsed = time.time()-t0
                seen = i*args.batch_size
                speed = seen/max(1e-9,elapsed)
                print(f"[Train][Ep {epoch:02d}] step {i:04d}/{steps:04d} "
                      f"loss={running/i:.4f} lr={sched.get_last_lr()[0]:.2e} speed~{speed:.1f} samp/s")

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
            Y = np.concatenate(all_true, axis=0).reshape(-1,8)
            P = np.concatenate(all_prob, axis=0).reshape(-1,8)
            metrics = compute_metrics_np(Y,P,thr=args.eval_threshold)
            print(f"[Val][Ep {epoch:02d}] bacc={metrics['bacc']:.4f} f1={metrics['f1']:.4f} auroc={metrics['auroc']}")
            if metrics["bacc"] > best_bacc:
                best_bacc = metrics["bacc"]
                torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
                print(f"  ✅ Saved best to {best_path}")

    print(f"Done. Best balanced accuracy={best_bacc:.4f}")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--win_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0, help="권장 0~2 (WSL/데스크탑 랙 방지)")
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--tcn_blocks", type=int, default=8, help="dilations=[1,2,4,...] 앞에서부터 사용")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--onset_weight", type=float, default=3.0)
    ap.add_argument("--tv_lambda", type=float, default=0.05)
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--feature_cache", type=int, default=128)
    ap.add_argument("--mmap", action="store_true", help="np.load(mmap_mode='r') 사용(초대형 파일용)")
    ap.add_argument("--compile", action="store_true", help="torch.compile 시도(초기 오버헤드 有)")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()
    train_main(args)

if __name__ == "__main__":
    main()

"""
python3 train_lasdra_fault_tcn_single.py \
  --data_dir data_storage/link_3 \
  --epochs 25 \
  --batch_size 24 \
  --win_size 256 \
  --workers 0 \
  --log_every 10
  """
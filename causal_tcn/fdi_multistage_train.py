# -*- coding: utf-8 -*-
"""
fdi_multistage_train.py

FDI_MultiStage (Cascaded, Temporal, Robust) — FULL TRAIN SCRIPT
- L1: onset detection (BiLSTM/CNN/Transformer 선택)
  * Focal BCE 선택 가능, val에서 임계치(θ) 스윕 + PR-AUC/지연(프레임) 로깅
- L2: temporal link-ID (강화된 상대/벡터 특성, causal trailing window)
  * 클래스 균형을 위한 Stratified 수집 + label smoothing
- L3~L5: temporal binary tree over motors (wrench observer + smoothing + longer window)
  * AUX: 8-way motor head + 트리-일관성(NLL) 손실
  * TimeMask/FeatDrop 증강, temporal TV 정칙(시퀀스 로짓 간 차분 규제)
- Curriculum: 성능 기반 Teacher-Forcing(라우팅), 일부 misroute 유지
- Labels: 레이아웃 자동 감지(grouped/interleaved) + fault-bit inversion sanity
- Samplers: L1 class-weighted, L2/L35는 stratified 클래스로 직접 생성
- Hard Negatives: 매 epoch val에서 FP/혼동 상위 세그먼트 수집→다음 epoch 학습에 주입
- Schedules: AdamW + Cosine decay with Warmup, GradClip, EMA(평가/저장 EMA 사용)
"""

import os, math, random, glob, gc, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset

# -------------------- MPS / 스레드 기본값 --------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ============================================================
#                   Switches / Hyperparams
# ============================================================
SEED               = 1234
DATA_ROOT          = "data_storage"
FILES_PER_LOAD     = 10
SAVE_DIR           = "FDI_MultiStage"
EPS_STD            = 1e-6
DT_DEFAULT         = 0.01
USE_GLOBAL_ZNORM   = True

# -------- L1 (fault-onset) --------
L1_MODEL_TYPE      = "bilstm"    # "bilstm" | "cnn1d" | "transformer"
L1_WINDOW          = 256
L1_POS_TOL         = 2
L1_THRESH_INIT     = 0.60
L1_HIDDEN          = 256
L1_LAYERS          = 2
L1_DROPOUT         = 0.30
L1_EPOCHS          = 30
L1_BS              = 64
L1_LR              = 3e-4
L1_MIN_LR          = 1e-5
L1_FOCAL           = True
L1_FOCAL_GAMMA     = 2.0
L1_FOCAL_ALPHA     = 0.5

# 메모리/샘플링
L1_NEG_PER_POS           = 3
L1_NEG_ONLY_PER_SEQ      = 5
L1_MAX_WINS_PER_SHARD    = 200_000

# -------- L2 (link id; TEMPORAL & RELATIVE) --------
L2_WINDOW          = 384
L2_EPOCHS          = 20
L2_BS              = 128
L2_LR              = 3e-4
L2_MIN_LR          = 1e-5
L2_HIDDEN          = 256
L2_LAYERS          = 2
L2_DROPOUT         = 0.20
L2_FEAT_DIM        = 28
CE_LABEL_SMOOTH    = 0.05

# -------- L3-5 (binary tree over motors; TEMPORAL) --------
L35_EPOCHS         = 15
L35_BS             = 128
L35_LR             = 3e-4
L35_MIN_LR         = 1e-5
L35_HIDDEN         = 128
L35_LAYERS         = 1
L35_DROPOUT        = 0.30
L35_WIN            = 512

# -------- Loss Weights / Regularizers -------
W_L1, W_L2, W_L35  = 1.0, 0.6, 0.4
LAMBDA_AUX8        = 0.5      # 8-way 보조헤드
LAMBDA_CONS        = 0.3      # 트리-8way 일관성(NLL) 손실
LAMBDA_TV          = 5e-5     # temporal TV 정칙(시퀀스 로짓 차분 L2)

# -------- Teacher Forcing / Routing -------
TF_ONSET_START = 0.70
TF_ONSET_END   = 0.10
WARMUP_L2_EPOCHS   = 4
WARMUP_L35_EPOCHS  = 12
MISROUTE_KEEP_P    = 0.20

# -------- Labels / Layout -------
LABEL_FAULT_IS_ONE_DEFAULT = True
FORCE_LABEL_LAYOUT = None            # None|"grouped"|"interleaved"

# -------- Stratify targets -------
FORCE_STRATIFY          = True
L2_MIN_TR_PER_CLASS     = 400
L2_MIN_VA_PER_CLASS     = 60
L2_MAX_TR_PER_CLASS     = 2000
L35_MIN_TR_PER_CLASS    = 400
L35_MIN_VA_PER_CLASS    = 60
L35_MAX_TR_PER_CLASS    = 2000
L35_FORCE_GT_SEGMENTS   = True
L35_ONSET_JITTER        = 12

# -------- Augment (TimeMask/FeatDrop) -------
AUG_TIME_MASK_P     = 0.5     # 배치에 적용 확률
AUG_TIME_MASK_MIN   = 10
AUG_TIME_MASK_MAX   = 30
AUG_FEAT_DROP_P     = 0.30
AUG_FEAT_DROP_FRAC  = 0.08    # 드랍 채널 비율

# -------- EMA & Scheduler -------
USE_EMA             = True
EMA_DECAY           = 0.999
WARMUP_FRAC         = 0.05     # 전체 epoch의 5% 워밍업
WEIGHT_DECAY        = 1e-3
GRAD_CLIP           = 1.0

# -------- Hard Negative Mining -------
HN_L2_MAX           = 5000
HN_8_MAX            = 5000
HN_MIX_FRAC         = 0.25     # 학습 풀에 섞는 비율(대략)

# -------- Z-Norm Sampling -------
ZNORM_MAX_SEQ      = 256
ZNORM_MAX_SAMPLES  = 250_000

# ============================================================
#                 Small utilities
# ============================================================
def set_all_seeds(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _to_cpu(sd): return {k: v.detach().cpu() for k,v in sd.items()}

# Cosine with Warmup (epoch-based)
class CosineWarmup:
    def __init__(self, optimizer, base_lrs, total_epochs, warmup_frac=WARMUP_FRAC, min_lrs=None):
        self.opt = optimizer
        self.base = list(base_lrs)
        self.total = total_epochs
        self.wu = max(1, int(round(total_epochs * warmup_frac)))
        self.min = list(min_lrs) if min_lrs is not None else [0.0]*len(base_lrs)
        assert len(self.opt.param_groups)==len(self.base)==len(self.min)
    def step_epoch(self, ep):
        if ep <= self.wu:
            scale = ep / float(self.wu)
        else:
            t = (ep - self.wu) / max(1, (self.total - self.wu))
            scale = 0.5*(1.0 + math.cos(math.pi * t))
        for pg, b, m in zip(self.opt.param_groups, self.base, self.min):
            pg['lr'] = m + (b - m) * scale

# Simple EMA manager (multi-module)
class EMAManager:
    def __init__(self, modules, decay=0.999):
        self.decay = decay
        self.shadow = []
        self.modules = list(modules)
        for md in self.modules:
            self.shadow.append({k: p.detach().clone() for k,p in md.state_dict().items()})
    @torch.no_grad()
    def update(self):
        for md, sh in zip(self.modules, self.shadow):
            sd = md.state_dict()
            for k in sd.keys():
                sh[k].mul_(self.decay).add_(sd[k]*(1.0-self.decay))
    @torch.no_grad()
    def copy_to(self):
        self._saved = [md.state_dict() for md in self.modules]
        for md, sh in zip(self.modules, self.shadow):
            md.load_state_dict(sh, strict=True)
    @torch.no_grad()
    def restore(self):
        for md, sav in zip(self.modules, self._saved):
            md.load_state_dict(sav, strict=True)

class use_ema:
    def __init__(self, ema: EMAManager):
        self.ema = ema
    def __enter__(self):
        if self.ema: self.ema.copy_to()
    def __exit__(self, exc_type, exc, tb):
        if self.ema: self.ema.restore()

# ============================================================
#                Basic SE(3) & Wrench utilities
# ============================================================
def _vee_skew(A: np.ndarray) -> np.ndarray:
    return np.stack([A[...,2,1]-A[...,1,2],
                     A[...,0,2]-A[...,2,0],
                     A[...,1,0]-A[...,0,1]], axis=-1).astype(np.float32) / 2.0

def _so3_log(Rm: np.ndarray) -> np.ndarray:
    Rm = Rm.astype(np.float32, copy=False)
    tr = np.clip((np.einsum('...ii', Rm)-1.0)/2.0, -1.0, 1.0).astype(np.float32)
    theta = np.arccos(tr).astype(np.float32)
    A = Rm - np.swapaxes(Rm, -1, -2)
    v = _vee_skew(A).astype(np.float32)
    sin_th = np.sin(theta).astype(np.float32)
    eps = np.float32(1e-9)
    scale = np.where(np.abs(sin_th)[...,None]>eps,
                     (theta/(sin_th+eps))[...,None],
                     1.0).astype(np.float32)
    w = (v*scale).astype(np.float32)
    return np.where((theta<1e-6)[...,None], v, w).astype(np.float32)

def finite_diff(x, dt):
    y = np.zeros_like(x, dtype=np.float32)
    y[...,1:,:] = (x[...,1:,:] - x[...,:-1,:]) / max(dt, 1e-6)
    return y.astype(np.float32)

def wrench_observer_discrete(alpha_k, alpha_km1, beta_k, Fext_km1, K1, K2, dt):
    return (Fext_km1 + K2 @ (alpha_k - alpha_km1 - beta_k*dt) - (K1 @ Fext_km1)*dt).astype(np.float32)

# ============================================================
#                    IO / Shards
# ============================================================
def load_10_npz_shards(link_count: int):
    data_dir = os.path.join(DATA_ROOT, f"link_{link_count}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    files = sorted([f for f in glob.glob(os.path.join(data_dir, "*.npz"))])
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files in {data_dir}")
    return files[:FILES_PER_LOAD]

class ShardFeatureCache:
    """
    Cache shard features on disk (per-shard .npy files) so that costly feature
    computation happens once while keeping runtime memory usage low via memmap.
    """
    def __init__(self, mu, std, cache_root):
        self.mu = None if mu is None else mu.astype(np.float32, copy=False)
        self.std = None if std is None else std.astype(np.float32, copy=False)
        self.cache_root = os.path.abspath(cache_root)
        os.makedirs(self.cache_root, exist_ok=True)
        self.cache = {}

    def _shard_dir(self, shard_path: str) -> str:
        base = os.path.splitext(os.path.basename(shard_path))[0]
        return os.path.join(self.cache_root, base)

    def _paths(self, shard_dir: str):
        return {
            "per_link_feat": os.path.join(shard_dir, "per_link_feat.npy"),
            "labels": os.path.join(shard_dir, "labels.npy"),
            "a_rel": os.path.join(shard_dir, "a_rel.npy"),
            "d_rel": os.path.join(shard_dir, "d_rel.npy"),
            "mass": os.path.join(shard_dir, "mass.npy"),
            "inertia": os.path.join(shard_dir, "inertia.npy"),
            "fu": os.path.join(shard_dir, "fu.npy"),
            "tau": os.path.join(shard_dir, "tau.npy"),
        }

    def _ensure_preprocessed(self, shard_path):
        shard_dir = self._shard_dir(shard_path)
        paths = self._paths(shard_dir)
        if all(os.path.exists(p) for p in paths.values()):
            return paths
        os.makedirs(shard_dir, exist_ok=True)
        dset = np.load(shard_path, allow_pickle=True)
        d_rel = dset["desired_link_rel"].astype(np.float32)
        a_rel = dset["actual_link_rel"].astype(np.float32)
        raw_lbl = dset["label"].astype(np.float32)
        lbl_inv = LABEL_FAULT_IS_ONE_DEFAULT
        if raw_lbl.shape[-1] >= 2:
            motors = raw_lbl[...,1:]
            lbl_inv = float((motors > 0.5).mean()) >= float((motors < 0.5).mean())
        labels = (1.0 - raw_lbl) if lbl_inv else raw_lbl
        per_link_feat = build_state_features(d_rel, a_rel, dt=DT_DEFAULT)
        if USE_GLOBAL_ZNORM and self.mu is not None and self.std is not None:
            per_link_feat = ((per_link_feat - self.mu) / self.std).astype(np.float32)
        masses = dset["mass"].astype(np.float32) if "mass" in dset else np.ones((d_rel.shape[2],), np.float32)
        inertias = dset["inertia"].astype(np.float32) if "inertia" in dset else np.tile(np.eye(3, dtype=np.float32)[None,...], (d_rel.shape[2],1,1))
        fu = dset["cmd_force"].astype(np.float32) if "cmd_force" in dset else np.zeros((d_rel.shape[0], d_rel.shape[1], d_rel.shape[2]), np.float32)
        tau = dset["cmd_torque"].astype(np.float32) if "cmd_torque" in dset else np.zeros((d_rel.shape[0], d_rel.shape[1], d_rel.shape[2],3), np.float32)
        per_link_feat = per_link_feat.astype(np.float32, copy=False)
        labels = labels.astype(np.float32, copy=False)
        fu = fu.astype(np.float32, copy=False)
        tau = tau.astype(np.float32, copy=False)
        np.save(paths["per_link_feat"], per_link_feat)
        np.save(paths["labels"], labels)
        np.save(paths["a_rel"], a_rel)
        np.save(paths["d_rel"], d_rel)
        np.save(paths["mass"], masses)
        np.save(paths["inertia"], inertias)
        np.save(paths["fu"], fu)
        np.save(paths["tau"], tau)
        del dset, d_rel, a_rel, raw_lbl, labels, per_link_feat, masses, inertias, fu, tau
        gc.collect()
        return paths

    def get(self, shard_path):
        entry = self.cache.get(shard_path)
        if entry is not None:
            return entry
        paths = self._ensure_preprocessed(shard_path)
        entry = {
            "per_link_feat": np.load(paths["per_link_feat"], mmap_mode='r'),
            "labels": np.load(paths["labels"], mmap_mode='r'),
            "a_rel": np.load(paths["a_rel"], mmap_mode='r'),
            "d_rel": np.load(paths["d_rel"], mmap_mode='r'),
            "mass": np.load(paths["mass"], mmap_mode='r'),
            "inertia": np.load(paths["inertia"], mmap_mode='r'),
            "fu": np.load(paths["fu"], mmap_mode='r'),
            "tau": np.load(paths["tau"], mmap_mode='r'),
        }
        self.cache[shard_path] = entry
        return entry

# ============================================================
#                 Feature builders
# ============================================================
def build_state_features(d_rel, a_rel, dt=DT_DEFAULT):
    d_rel = d_rel.astype(np.float32, copy=False)
    a_rel = a_rel.astype(np.float32, copy=False)
    S,T,L = d_rel.shape[:3]
    p_d = d_rel[...,:3,3]; p_a = a_rel[...,:3,3]
    R_d = d_rel[...,:3,:3]; R_a = a_rel[...,:3,:3]
    r_d = _so3_log(R_d);    r_a = _so3_log(R_a)

    v_d = finite_diff(p_d, dt); v_a = finite_diff(p_a, dt)
    a_d = finite_diff(v_d, dt); a_a = finite_diff(v_a, dt)
    omg_d = finite_diff(r_d, dt); omg_a = finite_diff(r_a, dt)

    pe = (p_d - p_a); ve = (v_d - v_a); ae = (a_d - a_a); oe = (omg_d - omg_a)

    per_link = np.concatenate([p_d, p_a, pe,
                               v_d, v_a, ve,
                               a_d, a_a, ae,
                               omg_d, omg_a, oe], axis=-1).astype(np.float32)  # (S,T,L,36)
    return per_link

# ============================================================
#         Label layout & inversion detection
# ============================================================
def _choose_link_mapping(per_link_feat, motors, onset_t, L):
    def salient_link_idx(seg_pe, seg_ve, seg_ae, seg_oe):
        n = np.linalg.norm(seg_pe,axis=-1) + np.linalg.norm(seg_ve,axis=-1) \
            + np.linalg.norm(seg_ae,axis=-1) + np.linalg.norm(seg_oe,axis=-1)
        return int(n.sum(axis=0).argmax())

    S,T,Lc,_ = per_link_feat.shape; assert Lc==L
    pe = per_link_feat[..., 6:9]; ve = per_link_feat[..., 15:18]
    ae = per_link_feat[..., 24:27]; oe = per_link_feat[..., 33:36]

    scoreA = 0; scoreB = 0; cnt=0
    for s in range(S):
        t0 = int(onset_t[s]); 
        if t0 < 0: continue
        t1 = max(0, t0-16); t2 = min(T, t0+1)
        sal = salient_link_idx(pe[s,t1:t2], ve[s,t1:t2], ae[s,t1:t2], oe[s,t1:t2])
        idx = int(motors[s,t0].argmax())
        linkA = (idx // 8); linkB = (idx % L)
        scoreA += (linkA==sal); scoreB += (linkB==sal); cnt += 1
    if cnt==0: return "grouped"
    return "grouped" if scoreA >= scoreB else "interleaved"

def extract_fault_targets(labels, L, layout="grouped"):
    MOTORS_PER_LINK = 8
    labels = labels.astype(np.float32, copy=False)
    S,T,M = labels.shape[:3]
    motors = labels[...,1:] if (M == 8*L + 1) else labels
    idx = motors.argmax(axis=2)       # (S,T)
    val = motors.max(axis=2)          # (S,T)
    fault_any = (val > 0.5)
    if layout=="grouped":
        link_from_idx = (idx // MOTORS_PER_LINK)
    else:
        link_from_idx = (idx % L)
    fault_motor = np.where(fault_any, idx, -1).astype(np.int64)
    fault_link  = np.where(fault_any, link_from_idx+1, 0).astype(np.int64)  # 1..L, 0: none
    onset_t = np.full((S,), -1, dtype=np.int64)
    for s in range(S):
        nz = np.where(fault_any[s])[0]
        if nz.size>0: onset_t[s] = int(nz[0])
    return fault_any, fault_link, fault_motor, onset_t

# ============================================================
#                  L1 dataset (윈도우)
# ============================================================
def build_l1_windows(per_link_feat, labels, L, win=L1_WINDOW, pos_tol=L1_POS_TOL,
                     neg_per_pos=L1_NEG_PER_POS, neg_only_per_seq=L1_NEG_ONLY_PER_SEQ,
                     max_wins_per_shard=L1_MAX_WINS_PER_SHARD, seed=SEED):
    rng = np.random.RandomState(seed)
    S,T,Lc,Dl = per_link_feat.shape; assert Lc == L
    _, _, _, onset_t = extract_fault_targets(labels, L, layout="grouped")  # 라벨 스캔만
    D_total = L * Dl
    ft = per_link_feat.reshape(S,T, D_total)

    X_list=[]; y_list=[]; sid_list=[]; tend_list=[]
    total_cap = max_wins_per_shard if (max_wins_per_shard is not None) else np.inf
    count = 0

    for s in range(S):
        t_candidates = list(range(T)); t0 = int(onset_t[s])
        if t0 >= 0:
            pos_idx = [t for t in t_candidates if abs(t - t0) <= pos_tol]
            neg_cands = [t for t in t_candidates if t not in pos_idx]
            num_neg = min(len(neg_cands), len(pos_idx)*neg_per_pos)
            neg_idx = rng.choice(neg_cands, size=num_neg, replace=False).tolist() if num_neg>0 else []
            select = sorted(pos_idx + neg_idx)
            for t in select:
                if count >= total_cap: break
                start_idx = max(0, t - win + 1)
                xw = ft[s, start_idx:t+1, :].astype(np.float32)
                if xw.shape[0] < win:
                    pad = np.repeat(xw[0:1], win - xw.shape[0], axis=0)
                    xw = np.concatenate([pad, xw], axis=0)
                y = 1 if abs(t - t0) <= pos_tol else 0
                X_list.append(xw); y_list.append(y)
                sid_list.append(s); tend_list.append(t); count += 1
        else:
            if neg_only_per_seq <= 0: continue
            neg_idx = rng.choice(t_candidates, size=min(len(t_candidates), neg_only_per_seq), replace=False).tolist()
            for t in neg_idx:
                if count >= total_cap: break
                start_idx = max(0, t - win + 1)
                xw = ft[s, start_idx:t+1, :].astype(np.float32)
                if xw.shape[0] < win:
                    pad = np.repeat(xw[0:1], win - xw.shape[0], axis=0)
                    xw = np.concatenate([pad, xw], axis=0)
                X_list.append(xw); y_list.append(0)
                sid_list.append(s); tend_list.append(t); count += 1
        if count >= total_cap: break

    if len(X_list)==0:
        return (np.zeros((0,win,D_total),dtype=np.float32),
                np.zeros((0,),dtype=np.int64),
                np.zeros((0,),dtype=np.int64),
                np.zeros((0,),dtype=np.int64),
                onset_t)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    seq_ids = np.array(sid_list, dtype=np.int64)
    t_end   = np.array(tend_list, dtype=np.int64)
    return X, y, seq_ids, t_end, onset_t

# ============================================================
#                      Models
# ============================================================
class L1BiLSTM(nn.Module):
    def __init__(self, in_dim, hidden=L1_HIDDEN, layers=L1_LAYERS, dropout=L1_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True,
                            dropout=dropout if layers>1 else 0.0, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*2, 1))
    def forward(self, x):  # (B,W,D)
        h,_ = self.lstm(x)
        out = self.head(h[:,-1,:])
        return out.squeeze(-1)

class L11DCNN(nn.Module):
    def __init__(self, in_dim, hidden=L1_HIDDEN, dropout=L1_DROPOUT):
        super().__init__()
        c = 64
        self.proj = nn.Linear(in_dim, c)
        self.conv1 = nn.Conv1d(c, 128, 5, padding=2)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 1))
    def forward(self, x):
        z = F.relu(self.proj(x))         # (B,W,C)
        z = z.transpose(1,2)             # (B,C,W)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))        # (B,128,W)
        z = z.mean(dim=2)                # GAP
        out = self.head(z)
        return out.squeeze(-1)

class L1Transformer(nn.Module):
    def __init__(self, in_dim, hidden=128, nhead=4, nlayers=2, dropout=L1_DROPOUT):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=hidden*4,
                                               dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
    def forward(self, x):
        z = self.in_proj(x)
        z = self.enc(z)
        z = z[:,-1,:]
        out = self.head(z)
        return out.squeeze(-1)

# ---- Temporal encoder / heads ----
class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=1, dropout=0.2, bidir=True, return_seq=False):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.rnn  = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=bidir, dropout=dropout if layers>1 else 0.0)
        self.out_dim = hidden * (2 if bidir else 1)
        self.return_seq = return_seq
    def forward(self, x):
        x = self.norm(x)
        h,_ = self.rnn(x)
        return h if self.return_seq else h[:,-1,:]  # (B,W,D) or (B,D)

class TemporalMultiClassHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim))
    def forward(self, z):
        return self.head(z)

class TemporalBinModel(nn.Module):
    def __init__(self, shared_encoder: TemporalEncoder, dropout=0.2):
        super().__init__()
        self.enc = shared_encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.enc.out_dim, 2))
    def forward(self, x):
        z = self.enc(x)
        if z.dim()==3: z = z[:,-1,:]    # 마지막 시점
        return self.head(z)

# ============================================================
#                 Losses / Metrics / Augment
# ============================================================
def bce_logits(pred, y):
    return F.binary_cross_entropy_with_logits(pred, y.float())

def focal_bce_with_logits(logits, targets, alpha=0.5, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    p_t = p*targets + (1-p)*(1-targets)
    loss = (alpha * (1 - p_t).pow(gamma) * ce).mean()
    return loss

def temporal_tv_on_logits(logits_seq):  # (B,W,C)
    if logits_seq.size(1) < 2: return logits_seq.new_tensor(0.0)
    dif = logits_seq[:,1:,:] - logits_seq[:,:-1,:]
    return (dif.pow(2).mean())

def pr_auc(scores, y_true):
    # 간단 PR-AUC (threshold sweep)
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y==1)
    fp = np.cumsum(y==0)
    prec = tp / np.maximum(tp+fp, 1)
    rec  = tp / max((y_true==1).sum(), 1)
    # R이 증가하도록 정렬되어 있음
    auc = 0.0
    for i in range(1, len(prec)):
        auc += (rec[i] - rec[i-1]) * prec[i]
    return float(auc)

def find_best_threshold(scores, y_true):
    best_th = L1_THRESH_INIT; best_f1 = -1.0
    for th in np.linspace(0.05, 0.95, 19):
        pred = (scores > th).astype(np.int64)
        tp = ((pred==1) & (y_true==1)).sum()
        fp = ((pred==1) & (y_true==0)).sum()
        fn = ((pred==0) & (y_true==1)).sum()
        prec = tp / max(tp+fp,1)
        rec  = tp / max(tp+fn,1)
        f1 = 2*prec*rec / max(prec+rec, 1e-9)
        if f1 > best_f1:
            best_f1 = f1; best_th = float(th)
    return best_th, best_f1

def onset_metrics_from_windows(sigmoid_scores, seq_ids, t_end, onset_t, theta, tol=5):
    S = int(seq_ids.max()) + 1 if seq_ids.size > 0 else 0
    pred_onset_local = np.full((S,), -1, dtype=np.int64)
    for s, t, q in zip(seq_ids, t_end, sigmoid_scores):
        if q > theta and pred_onset_local[s] == -1:
            pred_onset_local[s] = int(t)
    hits = 0; tot = 0; lats=[]
    for s in range(S):
        t_true = onset_t[s] if s < len(onset_t) else -1
        if t_true < 0: continue
        tot += 1
        t_pred = pred_onset_local[s]
        if t_pred >= 0:
            lat = t_pred - t_true
            lats.append(lat)
            if abs(lat) <= tol: hits += 1
    onset_acc = hits / max(tot, 1)
    mean_lat = np.mean(lats) if len(lats) > 0 else float('nan')
    return onset_acc, mean_lat

def apply_time_aug(xb, time_mask_p=AUG_TIME_MASK_P, feat_drop_p=AUG_FEAT_DROP_P,
                   time_min=AUG_TIME_MASK_MIN, time_max=AUG_TIME_MASK_MAX, feat_frac=AUG_FEAT_DROP_FRAC):
    # xb: (B,W,D)
    if xb.dim()!=3: return xb
    B,W,D = xb.shape
    if random.random() < time_mask_p:
        L = random.randint(time_min, min(time_max, W))
        t0 = random.randint(0, W-L)
        xb[:, t0:t0+L, :] = 0.0
    if random.random() < feat_drop_p:
        k = max(1, int(round(D*feat_frac)))
        dims = np.random.choice(D, size=k, replace=False)
        xb[:,:,dims] = 0.0
    return xb

# ============================================================
#                   Helpers: routing & pred
# ============================================================
def tf_linear_prob(ep, total, start_p, end_p):
    if total <= 1: return end_p
    r = max(0.0, min(1.0, 1.0 - (ep-1)/(total-1)))
    return end_p + (start_p - end_p) * r  # ep=1 -> start, ep=total -> end

def route_tfp_from_perf(ep, total_epochs, l2_val_acc):
    base_linear = max(0.3, 1.0 - (ep-1)/(total_epochs-1 + 1e-9))
    if l2_val_acc is None:
        return max(base_linear, 0.7)
    if l2_val_acc < 0.7: return max(0.9, base_linear)
    if l2_val_acc < 0.8: return max(0.7, base_linear)
    if l2_val_acc < 0.9: return max(0.5, base_linear*0.8)
    return 0.3

@torch.no_grad()
def l1_predict_onset_for_sequence(model, ft_seq, device, win=L1_WINDOW, theta=0.6, step=1, consec=2):
    model.eval()
    T = ft_seq.shape[0]
    if T < win: return -1
    windows, ends = [], []
    for t in range(win-1, T, step):
        windows.append(ft_seq[t-win+1:t+1,:]); ends.append(t)
    xb = torch.from_numpy(np.stack(windows,0)).float().to(device)
    q = torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy()
    run = 0
    for qi, t in zip(q, ends):
        if qi > theta:
            run += 1
            if run >= consec:
                return int(t)
        else:
            run = 0
    return -1

# ---- L2: temporal features (Causal trailing window, vector + relative) ----
def build_l2_seq_feats(per_link_feat_s, t_center, win=L2_WINDOW):
    T, L, _ = per_link_feat_s.shape
    t2 = min(T, t_center+1)
    t1 = max(0, t2 - win)
    pe = per_link_feat_s[..., 6:9]; ve = per_link_feat_s[..., 15:18]
    ae = per_link_feat_s[..., 24:27]; oe = per_link_feat_s[..., 33:36]
    seg_pe = pe[t1:t2]; seg_ve = ve[t1:t2]; seg_ae = ae[t1:t2]; seg_oe = oe[t1:t2]  # (W',L,3)
    Wseg = seg_pe.shape[0]

    # center across links (relative)
    pe_c = seg_pe - seg_pe.mean(axis=1, keepdims=True)
    ve_c = seg_ve - seg_ve.mean(axis=1, keepdims=True)
    ae_c = seg_ae - seg_ae.mean(axis=1, keepdims=True)
    oe_c = seg_oe - seg_oe.mean(axis=1, keepdims=True)

    def ratios(x):
        n = np.linalg.norm(x, axis=-1)  # (W',L)
        den = n.sum(axis=1, keepdims=True) + 1e-6
        return (n / den)[...,None]      # (W',L,1)
    r_pe = ratios(seg_pe); r_ve = ratios(seg_ve); r_ae = ratios(seg_ae); r_oe = ratios(seg_oe)

    feat = np.concatenate([
        pe_c, ve_c, ae_c, oe_c,          # centered residuals
        seg_pe, seg_ve, seg_ae, seg_oe,  # raw residuals
        r_pe, r_ve, r_ae, r_oe           # norm ratios
    ], axis=-1)  # (W',L,L2_FEAT_DIM)
    if Wseg < win:
        pad = np.zeros((win-Wseg, feat.shape[1], feat.shape[2]), dtype=np.float32)
        feat = np.concatenate([pad, feat], axis=0)
    return feat.reshape(win, feat.shape[1]*feat.shape[2]).astype(np.float32)

@torch.no_grad()
def l2_predict_link(model, x2_seq, device):
    model.eval()
    xb = torch.from_numpy(x2_seq[None,...]).float().to(device)
    out = model(xb).argmax(-1).item()
    return int(out)

# ---- L3~L5: temporal wrench features (smoothing + causal) ----
def ema_smooth(x, alpha=0.3):
    y = np.copy(x).astype(np.float32)
    for c in range(y.shape[-1]):
        acc = y[0, c]
        for t in range(1, y.shape[0]):
            acc = alpha*y[t, c] + (1.0-alpha)*acc
            y[t, c] = acc
    return y

def robust_clip(x, lo_q=5.0, hi_q=95.0):
    lo = np.percentile(x, lo_q, axis=0, keepdims=True)
    hi = np.percentile(x, hi_q, axis=0, keepdims=True)
    return np.clip(x, lo, hi)

def wrench_segment_for_link(
    m_i, I_i, v_i, w_i, fu_i, tau_i, R_i,
    dt, K1_i, K2_i, t1, t2, g=np.array([0,0,-9.81], dtype=np.float32)
):
    Tseg = t2 - t1
    Fseg = np.zeros((Tseg, 6), dtype=np.float32)
    alpha_prev = np.concatenate([m_i*v_i[t1], (I_i @ w_i[t1])], axis=0).astype(np.float32)
    Fprev = np.zeros(6, dtype=np.float32)
    if fu_i.ndim == 1:
        e3 = np.array([0,0,1], dtype=np.float32)
        fu_vec = (fu_i[:,None] * (R_i @ e3)).astype(np.float32)
    else:
        fu_vec = fu_i.astype(np.float32)
    for kk, k in enumerate(range(t1, t2)):
        alpha_k = np.concatenate([m_i*v_i[k], (I_i @ w_i[k])], axis=0).astype(np.float32)
        beta_k  = np.concatenate([
            (fu_vec[k] - m_i*g).astype(np.float32),
            (tau_i[k] - np.cross(w_i[k], (I_i @ w_i[k]).astype(np.float32))).astype(np.float32)
        ], axis=0).astype(np.float32)
        if k > t1:
            Fprev = wrench_observer_discrete(alpha_k, alpha_prev, beta_k, Fprev, K1_i, K2_i, dt)
        Fseg[kk] = Fprev
        alpha_prev = alpha_k
    return Fseg

def build_wrench_seq_feats_for_link_segment(dset_dict, link_idx, t_center, win=L35_WIN, dt=DT_DEFAULT):
    a_rel = dset_dict['a_rel'][0]; d_rel = dset_dict['d_rel'][0]
    masses = dset_dict['mass']; inertias = dset_dict['inertia']
    fu = dset_dict['fu']; tau = dset_dict['tau']

    T = a_rel.shape[0]
    t2 = min(T, t_center+1); t1 = max(0, t2 - win)

    p = a_rel[:, :, :3, 3]
    v_fd = np.zeros_like(p); v_fd[1:] = (p[1:] - p[:-1]) / max(dt,1e-6)
    rvec = _so3_log(a_rel[:, :, :3, :3])
    w_fd = np.zeros_like(rvec); w_fd[1:] = (rvec[1:] - rvec[:-1]) / max(dt,1e-6)

    m_i = float(masses[link_idx]); I_i = inertias[link_idx].astype(np.float32)
    v_i = v_fd[:, link_idx, :].astype(np.float32)
    w_i = w_fd[:, link_idx, :].astype(np.float32)
    R_i = a_rel[:, link_idx, :3,:3].astype(np.float32)

    if fu.ndim == 3: fu_i = fu[0,:,link_idx].astype(np.float32)       # (T,)
    else:            fu_i = fu[0,:,link_idx,:].astype(np.float32)
    tau_i = tau[0,:,link_idx,:].astype(np.float32)

    K1 = (np.eye(6, dtype=np.float32)*0.05).astype(np.float32)
    K2 = (np.eye(6, dtype=np.float32)*0.10).astype(np.float32)

    F_seg = wrench_segment_for_link(m_i, I_i, v_i, w_i, fu_i, tau_i, R_i, dt, K1, K2, t1, t2)  # (W',6)
    F_sm = ema_smooth(F_seg.T, alpha=0.3).T
    dF_seg = np.zeros_like(F_sm); dF_seg[1:] = (F_sm[1:] - F_sm[:-1]) / max(dt,1e-6)

    F_sm = robust_clip(F_sm, 5, 95); dF_seg = robust_clip(dF_seg, 5, 95)

    if fu.ndim == 3:
        thrust = fu[0, t1:t2, link_idx]
        u4_seg = np.stack([thrust, tau[0, t1:t2, link_idx, 0], tau[0, t1:t2, link_idx, 1], tau[0, t1:t2, link_idx, 2]], axis=-1).astype(np.float32)
    else:
        thrust = fu[0, t1:t2, link_idx, 2] if fu.shape[-1]==3 else fu[0, t1:t2, link_idx, 0]
        u4_seg = np.stack([thrust, tau[0, t1:t2, link_idx, 0], tau[0, t1:t2, link_idx, 1], tau[0, t1:t2, link_idx, 2]], axis=-1).astype(np.float32)

    feats_ts = np.concatenate([F_sm, dF_seg, u4_seg], axis=-1).astype(np.float32)  # (W',16)
    Wseg = feats_ts.shape[0]
    if Wseg < win:
        pad = np.zeros((win - Wseg, feats_ts.shape[1]), dtype=np.float32)
        feats_ts = np.concatenate([pad, feats_ts], axis=0)
    return feats_ts

# ============================================================
#      Stratified buckets & epoch-wise dynamic builders
# ============================================================
class StratifiedBuckets:
    def __init__(self, num_classes, seed=SEED):
        self.k = int(num_classes)
        self.X = [[] for _ in range(self.k)]
        self.y = [[] for _ in range(self.k)]
        self.rng = np.random.RandomState(seed)
    def add(self, x, y_cls):
        if 0 <= y_cls < self.k:
            self.X[y_cls].append(x); self.y[y_cls].append(y_cls)
    def _upsample(self, items, need):
        if len(items) == 0: return []
        idx = self.rng.randint(0, len(items), size=need)
        return [items[i] for i in idx]
    def _undersample(self, items, cap):
        if len(items) <= cap: return items
        idx = self.rng.choice(len(items), size=cap, replace=False)
        return [items[i] for i in idx]
    def finalize(self, min_tr, min_va, max_tr, va_frac=0.2, X_add=None, y_add=None, add_cap=None):
        X_tr_all, y_tr_all, X_va_all, y_va_all = [], [], [], []
        # optional: add hard negatives to train pool before per-class split
        if X_add is not None and y_add is not None and len(X_add)>0:
            # hard neg는 class-balanced가 아니므로, finalize 후 뒤에서 비율 섞는 대신
            # 간단히 전체 train에 concat
            pass  # we'll concat after class loops
        for c in range(self.k):
            Xc = self.X[c]; 
            idx = self.rng.permutation(len(Xc))
            Xc = [Xc[i] for i in idx]
            va_target = max(min_va, int(np.ceil(len(Xc) * va_frac)))
            Xc_va = self._upsample(Xc, va_target - len(Xc)) + Xc if len(Xc) < va_target else Xc[:va_target]
            Xc_tr_pool = Xc[va_target:]
            if len(Xc_tr_pool) < min_tr:
                base = Xc_tr_pool if len(Xc_tr_pool)>0 else Xc_va
                Xc_tr = Xc_tr_pool + self._upsample(base, min_tr - len(Xc_tr_pool))
            else:
                Xc_tr = Xc_tr_pool
            Xc_tr = self._undersample(Xc_tr, max_tr)
            X_tr_all.extend(Xc_tr); y_tr_all.extend([c]*len(Xc_tr))
            X_va_all.extend(Xc_va); y_va_all.extend([c]*len(Xc_va))
        if X_add is not None and y_add is not None and len(X_add)>0:
            if add_cap is not None and len(X_add) > add_cap:
                sel = self.rng.choice(len(X_add), size=add_cap, replace=False)
                X_add = X_add[sel]; y_add = y_add[sel]
            X_tr_all.extend(list(X_add)); y_tr_all.extend(list(y_add))
        X_tr = np.stack(X_tr_all, 0).astype(np.float32) if len(X_tr_all)>0 else None
        y_tr = np.array(y_tr_all, dtype=np.int64) if len(y_tr_all)>0 else None
        X_va = np.stack(X_va_all, 0).astype(np.float32) if len(X_va_all)>0 else None
        y_va = np.array(y_va_all, dtype=np.int64) if len(y_va_all)>0 else None
        return X_tr, y_tr, X_va, y_va

def _hist(y, k):
    if y is None: return "[]"
    bc = np.bincount(y, minlength=k)
    return "[" + ", ".join(str(int(c)) for c in bc) + "]"

def _jitter_center(rng, t, T, jitter=0):
    if jitter <= 0: return t
    return int(np.clip(t + rng.randint(-jitter, jitter+1), 0, T-1))

def make_balanced_loader(X, y, bs, is_val=False):
    if X is None or y is None or len(X)==0: return None
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y.astype(np.int64))
    if is_val:
        return DataLoader(TensorDataset(X_t, y_t), batch_size=bs, shuffle=False, num_workers=0)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=bs, shuffle=True, num_workers=0)

def build_dynamic_epoch_sets(
    shard_paths, device, L, l1_model, l2_model,
    znorm_mu, znorm_std,
    ep, total_epochs,
    warmup_l2=WARMUP_L2_EPOCHS, warmup_l35=WARMUP_L35_EPOCHS,
    l1_theta=0.6, l2_val_acc=None, label_layout=None,
    hn_l2=None, hn_8=None,
    shard_cache=None
):
    rng = np.random.RandomState(SEED+ep)
    tfp_onset = tf_linear_prob(ep, total_epochs, TF_ONSET_START, TF_ONSET_END)
    tfp_route = route_tfp_from_perf(ep, total_epochs, l2_val_acc)

    l2_buckets  = StratifiedBuckets(num_classes=L, seed=SEED+1000+ep)
    l35_buckets = StratifiedBuckets(num_classes=8, seed=SEED+2000+ep)
    route_try = 0; route_hit = 0

    for shard_path in shard_paths:
        if shard_cache is not None:
            entry = shard_cache.get(shard_path)
            per_link_feat = entry["per_link_feat"]
            labels = entry["labels"]
            a_rel = entry["a_rel"]
            d_rel = entry["d_rel"]
            masses = entry["mass"]
            inertias = entry["inertia"]
            fu = entry["fu"]
            tau = entry["tau"]
        else:
            dset = np.load(shard_path, allow_pickle=True)
            d_rel = dset["desired_link_rel"].astype(np.float32)
            a_rel = dset["actual_link_rel"].astype(np.float32)
            raw_lbl = dset["label"].astype(np.float32)

            lbl_inv = LABEL_FAULT_IS_ONE_DEFAULT
            if raw_lbl.shape[-1] >= 2:
                motors = raw_lbl[...,1:]
                lbl_inv = float((motors > 0.5).mean()) >= float((motors < 0.5).mean())
            labels = (1.0 - raw_lbl) if lbl_inv else raw_lbl

            per_link_feat = build_state_features(d_rel, a_rel, dt=DT_DEFAULT)
            if USE_GLOBAL_ZNORM:
                per_link_feat = ((per_link_feat - znorm_mu) / znorm_std).astype(np.float32)
            masses = dset["mass"].astype(np.float32) if "mass" in dset else np.ones((L,),np.float32)
            inertias = dset["inertia"].astype(np.float32) if "inertia" in dset else np.tile(np.eye(3,dtype=np.float32)[None,...], (L,1,1))
            fu = dset["cmd_force"].astype(np.float32) if "cmd_force" in dset else np.zeros((d_rel.shape[0], d_rel.shape[1], L),np.float32)
            tau = dset["cmd_torque"].astype(np.float32) if "cmd_torque" in dset else np.zeros((d_rel.shape[0], d_rel.shape[1], L,3),np.float32)

        S,T,L_local = d_rel.shape[:3]; assert L_local == L

        if (label_layout is None) and (ep == 1) and (shard_path == shard_paths[0]):
            motors = labels[...,1:] if labels.shape[-1]==8*L+1 else labels
            _, _, _, onset_tmp = extract_fault_targets(labels, L, "grouped")
            label_layout = _choose_link_mapping(per_link_feat, motors, onset_tmp, L)
        if label_layout is None:
            label_layout = "grouped"

        fault_any, fault_link, fault_motor, onset_gt = extract_fault_targets(labels, L, layout=label_layout)

        for s in range(S):
            # 1) onset
            use_gt_onset = (rng.rand() < tfp_onset) or (ep <= 2)
            t_use = int(onset_gt[s]) if use_gt_onset else -1
            if t_use < 0:
                ft_seq = per_link_feat[s].reshape(T, L*36)
                t_pred = l1_predict_onset_for_sequence(
                    l1_model, ft_seq, device, win=L1_WINDOW, theta=l1_theta, step=1, consec=2
                )
                t_use = int(t_pred)
                if t_use < 0:  # 온셋 실패
                    continue

            # 2) L2 표본
            x2_seq = build_l2_seq_feats(per_link_feat[s], t_use, win=L2_WINDOW)
            y2 = int(fault_link[s, t_use] - 1) if fault_link[s,t_use] > 0 else -1
            if y2 < 0 and onset_gt[s] >= 0:
                y2 = int(fault_link[s, onset_gt[s]] - 1)
            if not (0 <= y2 < L): continue
            l2_buckets.add(x2_seq, y2)

            # 3) 라우팅
            use_gt_route = (rng.rand() < tfp_route) or (l2_model is None) or (ep <= warmup_l2)
            link_sel = y2 if use_gt_route else l2_predict_link(l2_model, x2_seq, device)
            route_try += 1; route_ok = (link_sel == y2)
            if route_ok: route_hit += 1

            # 4) L35 표본 — GT 링크 기준 확보
            link_for_l35 = y2 if L35_FORCE_GT_SEGMENTS else link_sel
            t_center = _jitter_center(rng, t_use, T, jitter=L35_ONSET_JITTER)
            seq_view = {'a_rel': a_rel[s:s+1], 'd_rel': d_rel[s:s+1],
                        'mass': masses, 'inertia': inertias,
                        'fu': fu[s:s+1], 'tau': tau[s:s+1]}
            x35_seq = build_wrench_seq_feats_for_link_segment(seq_view, link_for_l35, t_center, win=L35_WIN, dt=DT_DEFAULT)
            if onset_gt[s] >= 0:
                m_true_global = int(np.argmax(labels[s, onset_gt[s], 1:]))
                if label_layout == "grouped": y_motor_in_link = (m_true_global % 8)
                elif label_layout == "interleaved": y_motor_in_link = (m_true_global // L)
                else: y_motor_in_link = (m_true_global % 8)
                l35_buckets.add(x35_seq, int(y_motor_in_link))

        if shard_cache is None:
            del dset, d_rel, a_rel, labels, per_link_feat, masses, inertias, fu, tau
        else:
            del per_link_feat, labels, a_rel, d_rel, masses, inertias, fu, tau, entry
        gc.collect()

    # ---------- finalize & add hard negatives ----------
    X2_tr, y2_tr, X2_va, y2_va = l2_buckets.finalize(
        min_tr=L2_MIN_TR_PER_CLASS, min_va=L2_MIN_VA_PER_CLASS,
        max_tr=L2_MAX_TR_PER_CLASS, va_frac=0.2,
        X_add=None if hn_l2 is None else hn_l2[0],
        y_add=None if hn_l2 is None else hn_l2[1],
        add_cap=int(HN_L2_MAX*HN_MIX_FRAC)
    )
    X35_tr, y35_tr, X35_va, y35_va = l35_buckets.finalize(
        min_tr=L35_MIN_TR_PER_CLASS, min_va=L35_MIN_VA_PER_CLASS,
        max_tr=L35_MAX_TR_PER_CLASS, va_frac=0.2,
        X_add=None if hn_8 is None else hn_8[0],
        y_add=None if hn_8 is None else hn_8[1],
        add_cap=int(HN_8_MAX*HN_MIX_FRAC)
    )

    # loaders
    dl2_tr  = make_balanced_loader(X2_tr,  y2_tr,  L2_BS,  is_val=False)
    dl2_va  = make_balanced_loader(X2_va,  y2_va,  L2_BS,  is_val=True)

    # tree split
    def split_tree_arrays_for(X_all, y_all, is_val):
        if X_all is None or y_all is None: 
            return (None,)*7
        def mk_loader(X, y):
            return make_balanced_loader(X, y, L35_BS, is_val=is_val)
        y3 = (y_all >= 4).astype(np.int64)  # L3
        l3_ld = mk_loader(X_all, y3)
        selA = (y_all < 4); selB = ~selA
        y4A = np.where(selA, (y_all//2), -1).astype(np.int64)
        y4B = np.where(selB, ((y_all-4)//2), -1).astype(np.int64)
        l4A_ld = mk_loader(X_all[selA], y4A[selA]) if selA.any() else None
        l4B_ld = mk_loader(X_all[selB], y4B[selB]) if selB.any() else None
        def pair_mask(y, a, b):
            m = (y==a) | (y==b); lab = np.where(y==a, 0, np.where(y==b,1,-1))
            return m, lab.astype(np.int64)
        m01, y5_01 = pair_mask(y_all, 0,1)
        m23, y5_23 = pair_mask(y_all, 2,3)
        m45, y5_45 = pair_mask(y_all, 4,5)
        m67, y5_67 = pair_mask(y_all, 6,7)
        l5_01_ld = mk_loader(X_all[m01], y5_01[m01]) if m01.any() else None
        l5_23_ld = mk_loader(X_all[m23], y5_23[m23]) if m23.any() else None
        l5_45_ld = mk_loader(X_all[m45], y5_45[m45]) if m45.any() else None
        l5_67_ld = mk_loader(X_all[m67], y5_67[m67]) if m67.any() else None
        return l3_ld, l4A_ld, l4B_ld, l5_01_ld, l5_23_ld, l5_45_ld, l5_67_ld

    (dl3_tr, dl4A_tr, dl4B_tr, dl5_01_tr, dl5_23_tr, dl5_45_tr, dl5_67_tr) = split_tree_arrays_for(X35_tr, y35_tr, is_val=False)
    (dl3_va, dl4A_va, dl4B_va, dl5_01_va, dl5_23_va, dl5_45_va, dl5_67_va) = split_tree_arrays_for(X35_va, y35_va, is_val=True)

    route_rate = float(route_hit / max(route_try,1))
    return (dl2_tr, dl2_va,
            (dl3_tr, dl4A_tr, dl4B_tr, dl5_01_tr, dl5_23_tr, dl5_45_tr, dl5_67_tr),
            (dl3_va, dl4A_va, dl4B_va, dl5_01_va, dl5_23_va, dl5_45_va, dl5_67_va),
            (X35_va, y35_va),
            (X35_tr, y35_tr),
            (X2_va, y2_va),
            route_rate, label_layout)

# ============================================================
#   Compose tree log-probs -> 8-way logits (for consistency)
# ============================================================
def compose_tree_logp_from_heads(z, l3_head, l4A_head, l4B_head, l5_01_head, l5_23_head, l5_45_head, l5_67_head):
    logp3   = F.log_softmax(l3_head(z), dim=1)
    log4A   = F.log_softmax(l4A_head(z), dim=1)
    log4B   = F.log_softmax(l4B_head(z), dim=1)
    log5_01 = F.log_softmax(l5_01_head(z), dim=1)
    log5_23 = F.log_softmax(l5_23_head(z), dim=1)
    log5_45 = F.log_softmax(l5_45_head(z), dim=1)
    log5_67 = F.log_softmax(l5_67_head(z), dim=1)
    c0 = logp3[:,0] + log4A[:,0] + log5_01[:,0]
    c1 = logp3[:,0] + log4A[:,0] + log5_01[:,1]
    c2 = logp3[:,0] + log4A[:,1] + log5_23[:,0]
    c3 = logp3[:,0] + log4A[:,1] + log5_23[:,1]
    c4 = logp3[:,1] + log4B[:,0] + log5_45[:,0]
    c5 = logp3[:,1] + log4B[:,0] + log5_45[:,1]
    c6 = logp3[:,1] + log4B[:,1] + log5_67[:,0]
    c7 = logp3[:,1] + log4B[:,1] + log5_67[:,1]
    return torch.stack([c0,c1,c2,c3,c4,c5,c6,c7], dim=1)  # (B,8)

# ============================================================
#                       Main
# ============================================================
def main():
    # device
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print("Using device:", device)
    set_all_seeds(SEED)
    pin_mem = (device.type == "cuda")

    # link count
    try:
        link_count = int(input("How many links?: ").strip())
    except Exception:
        link_count = 1; print("Invalid input. Using link_count=1.")

    # shards
    shard_paths = load_10_npz_shards(link_count)
    print(f"Loading {len(shard_paths)} shards...")

    # -------- Global Z-Norm --------
    dset0 = np.load(shard_paths[0], allow_pickle=True)
    d_rel0 = dset0["desired_link_rel"].astype(np.float32)
    a_rel0 = dset0["actual_link_rel"].astype(np.float32)
    L = d_rel0.shape[2]
    if USE_GLOBAL_ZNORM:
        print("  🧮 Precomputing global Z-Norm (sampled) ...")
        mu, std = compute_znorm_sampled(d_rel0, a_rel0, dt=DT_DEFAULT)
    else:
        mu = np.zeros((36,), dtype=np.float32); std = np.ones((36,), dtype=np.float32)
    del dset0, d_rel0, a_rel0; gc.collect()
    if device.type == "mps":
        try: torch.mps.empty_cache()
        except Exception: pass
    cache_dir = os.path.join(SAVE_DIR, "_shard_cache", f"link_{link_count}")
    shard_cache = ShardFeatureCache(mu if USE_GLOBAL_ZNORM else None,
                                    std if USE_GLOBAL_ZNORM else None,
                                    cache_root=cache_dir)

    # -------- L1 dataset --------
    print("🧱 Building L1 windows shard-by-shard ...")
    X1_list, y1_list, seq_list, tend_list, onset_all = [], [], [], [], []
    seq_offset = 0
    for i, shard_path in enumerate(shard_paths):
        print(f"  🔹 Shard {i+1}/{len(shard_paths)} → {os.path.basename(shard_path)}")
        if shard_cache is not None:
            entry = shard_cache.get(shard_path)
            per_link_feat = np.asarray(entry["per_link_feat"])
            labels = np.asarray(entry["labels"])
            d_rel = np.asarray(entry["d_rel"])
        else:
            dset = np.load(shard_path, allow_pickle=True)
            d_rel = dset["desired_link_rel"].astype(np.float32)
            a_rel = dset["actual_link_rel"].astype(np.float32)
            raw_lbl = dset["label"].astype(np.float32)
            lbl_inv = LABEL_FAULT_IS_ONE_DEFAULT
            if raw_lbl.shape[-1] >= 2:
                motors = raw_lbl[...,1:]
                lbl_inv = float((motors > 0.5).mean()) >= float((motors < 0.5).mean())
            labels = (1.0 - raw_lbl) if lbl_inv else raw_lbl
            per_link_feat = build_state_features(d_rel, a_rel, dt=DT_DEFAULT)
            if USE_GLOBAL_ZNORM:
                per_link_feat = ((per_link_feat - mu) / std).astype(np.float32)
        X1_s, y1_s, seq_ids_s, t_end_s, onset_s = build_l1_windows(
            per_link_feat, labels, L, win=L1_WINDOW, pos_tol=L1_POS_TOL,
            neg_per_pos=L1_NEG_PER_POS, neg_only_per_seq=L1_NEG_ONLY_PER_SEQ,
            max_wins_per_shard=L1_MAX_WINS_PER_SHARD, seed=SEED+i
        )
        X1_list.append(X1_s); y1_list.append(y1_s)
        seq_list.append(seq_ids_s + seq_offset)
        tend_list.append(t_end_s); onset_all.append(onset_s)
        seq_offset += d_rel.shape[0]
        if shard_cache is None:
            del dset, d_rel, a_rel, labels, per_link_feat
        else:
            del d_rel, per_link_feat, labels
        del X1_s, y1_s
        gc.collect()
        if device.type == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    X1 = np.concatenate(X1_list, 0) if len(X1_list)>0 else np.zeros((0,L1_WINDOW, L*36), np.float32)
    y1 = np.concatenate(y1_list, 0) if len(y1_list)>0 else np.zeros((0,), np.int64)
    seq_ids = np.concatenate(seq_list, 0) if len(seq_list)>0 else np.zeros((0,), np.int64)
    t_end = np.concatenate(tend_list, 0) if len(tend_list)>0 else np.zeros((0,), np.int64)
    onset_arr = np.concatenate(onset_all, 0) if len(onset_all)>0 else np.zeros((0,), np.int64)
    del X1_list, y1_list, seq_list, tend_list, onset_all; gc.collect()
    print(f"  ✅ Built Level1 dataset: {len(X1)} windows")
    if len(X1)==0: raise RuntimeError("L1 dataset empty.")

    # L1 loaders
    N1, W, D1 = X1.shape
    idx = np.random.RandomState(SEED).permutation(N1)
    n_tr1 = int(0.8*N1)
    tr_idx1, va_idx1 = idx[:n_tr1], idx[n_tr1:]
    X1_tr = torch.from_numpy(X1[tr_idx1]); y1_tr = torch.from_numpy(y1[tr_idx1])
    X1_va = torch.from_numpy(X1[va_idx1]); y1_va = torch.from_numpy(y1[va_idx1])
    w_up = 1.0 + 5.0*y1_tr.cpu().numpy()
    sampler = WeightedRandomSampler(weights=torch.from_numpy(w_up.astype(np.float32)),
                                    num_samples=len(w_up), replacement=True)
    dl1_tr = DataLoader(TensorDataset(X1_tr, y1_tr), batch_size=L1_BS, sampler=sampler, shuffle=False, pin_memory=pin_mem, num_workers=0)
    dl1_va = DataLoader(TensorDataset(X1_va, y1_va), batch_size=L1_BS, shuffle=False, pin_memory=pin_mem, num_workers=0)

    # =======================================================
    # Models
    # =======================================================
    if L1_MODEL_TYPE.lower()=="bilstm":
        l1_model = L1BiLSTM(D1).to(device)
    elif L1_MODEL_TYPE.lower()=="cnn1d":
        l1_model = L11DCNN(D1).to(device)
    elif L1_MODEL_TYPE.lower()=="transformer":
        l1_model = L1Transformer(D1).to(device)
    else:
        raise ValueError("L1_MODEL_TYPE must be one of {'bilstm','cnn1d','transformer'}")

    # L2 temporal
    l2_enc  = TemporalEncoder(in_dim=L*L2_FEAT_DIM, hidden=L2_HIDDEN, layers=L2_LAYERS, dropout=L2_DROPOUT, bidir=True).to(device)
    l2_head = TemporalMultiClassHead(in_dim=l2_enc.out_dim, out_dim=L, dropout=L2_DROPOUT).to(device)
    class L2Temporal(nn.Module):
        def __init__(self, enc, head):
            super().__init__(); self.enc=enc; self.head=head
        def forward(self, x):
            z = self.enc(x)
            return self.head(z)
    l2_model = L2Temporal(l2_enc, l2_head).to(device)

    # L3–L5 encoder (return seq for TV) + heads
    l35_enc = TemporalEncoder(in_dim=16, hidden=L35_HIDDEN, layers=L35_LAYERS, dropout=L35_DROPOUT, bidir=True, return_seq=True).to(device)
    l3_model   = TemporalBinModel(l35_enc, dropout=L35_DROPOUT).to(device)
    l4A_model  = TemporalBinModel(l35_enc, dropout=L35_DROPOUT).to(device)
    l4B_model  = TemporalBinModel(l35_enc, dropout=L35_DROPOUT).to(device)
    l5_01      = TemporalBinModel(l35_enc, dropout=L35_DROPOUT).to(device)
    l5_23      = TemporalBinModel(l35_enc, dropout=L35_DROPOUT).to(device)
    l5_45      = TemporalBinModel(l35_enc, dropout=L35_DROPOUT).to(device)
    l5_67      = TemporalBinModel(l35_enc, dropout=L35_DROPOUT).to(device)
    motor8_head = TemporalMultiClassHead(in_dim=l35_enc.out_dim, out_dim=8, dropout=L35_DROPOUT).to(device)

    # =======================================================
    # Optimizer / Scheduler / EMA
    # =======================================================
    param_groups = [
        {"params": l1_model.parameters(), "lr": L1_LR},
        {"params": l2_enc.parameters(),    "lr": L2_LR},
        {"params": l2_head.parameters(),   "lr": L2_LR},
        {"params": l35_enc.parameters(),   "lr": L35_LR},
        {"params": l3_model.head.parameters(),   "lr": L35_LR},
        {"params": l4A_model.head.parameters(),  "lr": L35_LR},
        {"params": l4B_model.head.parameters(),  "lr": L35_LR},
        {"params": l5_01.head.parameters(),      "lr": L35_LR},
        {"params": l5_23.head.parameters(),      "lr": L35_LR},
        {"params": l5_45.head.parameters(),      "lr": L35_LR},
        {"params": l5_67.head.parameters(),      "lr": L35_LR},
        {"params": motor8_head.parameters(),     "lr": L35_LR},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    min_lrs  = [min(L1_MIN_LR, L2_MIN_LR, L35_MIN_LR)]*len(base_lrs)
    TOTAL_EPOCHS = max(L1_EPOCHS, L2_EPOCHS, L35_EPOCHS)
    scheduler = CosineWarmup(optimizer, base_lrs, total_epochs=TOTAL_EPOCHS, warmup_frac=WARMUP_FRAC, min_lrs=min_lrs)
    ema = EMAManager([l1_model, l2_model, l35_enc, l3_model, l4A_model, l4B_model, l5_01, l5_23, l5_45, l5_67, motor8_head], decay=EMA_DECAY) if USE_EMA else None

    ckpt_dir = os.path.join(SAVE_DIR, f"link_{L}")
    os.makedirs(ckpt_dir, exist_ok=True)
    COMBINED_CKPT_PATH = os.path.join(ckpt_dir, "FDI_MultiStage_ALL.pth")
    best_val_combo = float('inf')
    l1_theta_runtime = L1_THRESH_INIT
    last_l2_val_acc = None
    chosen_label_layout = FORCE_LABEL_LAYOUT

    # Hard negative buffers
    HN_L2_X = np.zeros((0, L2_WINDOW, L*L2_FEAT_DIM), dtype=np.float32)
    HN_L2_y = np.zeros((0,), dtype=np.int64)
    HN_8_X  = np.zeros((0, L35_WIN, 16), dtype=np.float32)
    HN_8_y  = np.zeros((0,), dtype=np.int64)

    # ----------------------------- Epoch Loop -----------------------------
    for ep in range(1, TOTAL_EPOCHS+1):
        scheduler.step_epoch(ep)

        # ---------------- L1 Train ----------------
        l1_model.train(); tr_l1_loss = 0; n1=0
        if ep <= L1_EPOCHS:
            for xb,yb in dl1_tr:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = l1_model(xb)
                if L1_FOCAL:
                    loss = focal_bce_with_logits(logits, yb, alpha=L1_FOCAL_ALPHA, gamma=L1_FOCAL_GAMMA)
                else:
                    loss = bce_logits(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(l1_model.parameters(), GRAD_CLIP)
                optimizer.step()
                if ema: ema.update()
                tr_l1_loss += loss.item()*xb.size(0); n1 += xb.size(0)
        tr_l1_loss = tr_l1_loss / max(n1,1) if n1>0 else None

        # --------- Build epoch dynamic sets (with HN) ---------
        # HN 섞기 (일부만 사용)
        hn_l2_tuple = None
        if len(HN_L2_X)>0:
            cap = min(HN_L2_MAX, len(HN_L2_X))
            sel = np.random.choice(len(HN_L2_X), size=cap, replace=False)
            hn_l2_tuple = (HN_L2_X[sel], HN_L2_y[sel])
        hn_8_tuple = None
        if len(HN_8_X)>0:
            cap = min(HN_8_MAX, len(HN_8_X))
            sel = np.random.choice(len(HN_8_X), size=cap, replace=False)
            hn_8_tuple = (HN_8_X[sel], HN_8_y[sel])

        (dl2_tr, dl2_va,
         (dl3_tr, dl4A_tr, dl4B_tr, dl5_01_tr, dl5_23_tr, dl5_45_tr, dl5_67_tr),
         (dl3_va, dl4A_va, dl4B_va, dl5_01_va, dl5_23_va, dl5_45_va, dl5_67_va),
         (X35_va_base, y35_va_base),
         (X35_tr_base, y35_tr_base),
         (X2_va_base,  y2_va_base),
         route_rate, chosen_label_layout) = build_dynamic_epoch_sets(
            shard_paths, device, L, l1_model, l2_model, mu, std, ep, TOTAL_EPOCHS,
            warmup_l2=WARMUP_L2_EPOCHS, warmup_l35=WARMUP_L35_EPOCHS,
            l1_theta=l1_theta_runtime, l2_val_acc=last_l2_val_acc, label_layout=chosen_label_layout,
            hn_l2=hn_l2_tuple, hn_8=hn_8_tuple,
            shard_cache=shard_cache
        )

        # ---------------- L2 Train ----------------
        l2_model.train(); tr_l2_loss=0; n2=0
        if ep <= L2_EPOCHS and dl2_tr is not None:
            for xb,yb in dl2_tr:
                xb, yb = xb.to(device), yb.to(device)
                xb = apply_time_aug(xb)  # 증강
                optimizer.zero_grad(set_to_none=True)
                logits = l2_model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=CE_LABEL_SMOOTH)
                loss.backward()
                nn.utils.clip_grad_norm_(list(l2_model.parameters()), GRAD_CLIP)
                optimizer.step()
                if ema: ema.update()
                tr_l2_loss += loss.item()*xb.size(0); n2 += xb.size(0)
        tr_l2_loss = tr_l2_loss / max(n2,1) if n2>0 else None

        # ---------------- L3–L5 Train (bin) + AUX/CONS + TV ----------------
        def train_bin_epoch(dl, model):
            if dl is None: return None
            model.train(); loss_sum=0; n=0
            for xb,yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                xb = apply_time_aug(xb)
                optimizer.zero_grad(set_to_none=True)
                # enc는 시퀀스 반환 -> 마지막시점 로짓으로 CE, 시퀀스로 TV
                z_seq = l35_enc(xb)                 # (B,W,D)
                z_last = z_seq[:,-1,:]              # (B,D)
                logits = model.head(z_last)
                loss_ce = F.cross_entropy(logits, yb, label_smoothing=CE_LABEL_SMOOTH)
                # shared TV (8-way와 공유될 시퀀스 정칙)
                loss_tv = temporal_tv_on_logits(z_seq) * LAMBDA_TV
                loss = loss_ce + loss_tv
                loss.backward()
                nn.utils.clip_grad_norm_(list(l35_enc.parameters()) + list(model.head.parameters()), GRAD_CLIP)
                optimizer.step()
                if ema: ema.update()
                loss_sum += loss.item()*xb.size(0); n += xb.size(0)
            return (loss_sum / max(n,1)) if n>0 else None

        tr_l3 = tr_l4A = tr_l4B = tr_l5_01 = tr_l5_23 = tr_l5_45 = tr_l5_67 = None
        if ep <= L35_EPOCHS:
            tr_l3    = train_bin_epoch(dl3_tr,   l3_model)
            tr_l4A   = train_bin_epoch(dl4A_tr,  l4A_model)
            tr_l4B   = train_bin_epoch(dl4B_tr,  l4B_model)
            tr_l5_01 = train_bin_epoch(dl5_01_tr, l5_01)
            tr_l5_23 = train_bin_epoch(dl5_23_tr, l5_23)
            tr_l5_45 = train_bin_epoch(dl5_45_tr, l5_45)
            tr_l5_67 = train_bin_epoch(dl5_67_tr, l5_67)

        # AUX 8-way + CONS (tree) + TV
        tr_aux8 = tr_cons = None
        if ep <= L35_EPOCHS and X35_tr_base is not None and len(X35_tr_base)>0:
            aux_dl = make_balanced_loader(X35_tr_base, y35_tr_base, L35_BS, is_val=False)
            loss_sum8=0; loss_sum_cons=0; n=0
            for xb, yb in aux_dl:
                xb, yb = xb.to(device), yb.to(device)
                xb = apply_time_aug(xb)
                optimizer.zero_grad(set_to_none=True)
                z_seq = l35_enc(xb)              # (B,W,D)
                z_last = z_seq[:,-1,:]
                logits8 = motor8_head(z_last)
                loss8 = F.cross_entropy(logits8, yb, label_smoothing=CE_LABEL_SMOOTH)
                tree_logp8 = compose_tree_logp_from_heads(
                    z_last, l3_model.head, l4A_model.head, l4B_model.head, l5_01.head, l5_23.head, l5_45.head, l5_67.head
                )
                loss_cons = F.nll_loss(tree_logp8, yb)
                loss_tv = temporal_tv_on_logits(z_seq) * LAMBDA_TV
                loss = LAMBDA_AUX8*loss8 + LAMBDA_CONS*loss_cons + loss_tv
                loss.backward()
                nn.utils.clip_grad_norm_(list(l35_enc.parameters()) +
                                         list(motor8_head.parameters()) +
                                         list(l3_model.head.parameters()) +
                                         list(l4A_model.head.parameters()) +
                                         list(l4B_model.head.parameters()) +
                                         list(l5_01.head.parameters()) +
                                         list(l5_23.head.parameters()) +
                                         list(l5_45.head.parameters()) +
                                         list(l5_67.head.parameters()), GRAD_CLIP)
                optimizer.step()
                if ema: ema.update()
                bs = xb.size(0)
                loss_sum8 += loss8.item()*bs; loss_sum_cons += loss_cons.item()*bs; n += bs
            if n>0:
                tr_aux8 = loss_sum8/n; tr_cons = loss_sum_cons/n

        # ---------------------- VALID (EMA 가중치로) ----------------------
        if ema: 
            ctx = use_ema(ema)
        else:   
            class _N: 
                    def __enter__(self): pass
                    def __exit__(self, a,b,c): pass
            ctx = _N()
        with ctx:
            # L1
            l1_va_loss=None; l1_win_acc=l1_prec=l1_rec=l1_onset_acc=None; l1_lat=None; l1_prauc=None
            if ep <= L1_EPOCHS:
                l1_model.eval(); va_loss=0; n=0; all_scores=[]; all_y=[]
                with torch.no_grad():
                    for xb,yb in dl1_va:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = l1_model(xb)
                        loss = focal_bce_with_logits(logits, yb, alpha=L1_FOCAL_ALPHA, gamma=L1_FOCAL_GAMMA) if L1_FOCAL else bce_logits(logits, yb)
                        va_loss += loss.item()*xb.size(0); n += xb.size(0)
                        all_scores.append(torch.sigmoid(logits).cpu().numpy())
                        all_y.append(yb.cpu().numpy())
                if n>0:
                    l1_va_loss = va_loss/n
                    scores = np.concatenate(all_scores, axis=0); yv = np.concatenate(all_y, axis=0)
                    best_th, best_f1 = find_best_threshold(scores, yv)
                    l1_theta_runtime = best_th
                    pred_win = (scores > best_th).astype(np.int64)
                    l1_win_acc = float((pred_win==yv).mean())
                    tp = ((pred_win==1) & (yv==1)).sum()
                    l1_prec = float(tp / max(pred_win.sum(),1))
                    l1_rec  = float(tp / max((yv==1).sum(),1))
                    va_seq = seq_ids[va_idx1]; va_tend = t_end[va_idx1]
                    onset_acc, mean_lat = onset_metrics_from_windows(scores, va_seq, va_tend, onset_arr, theta=best_th, tol=5)
                    l1_onset_acc = float(onset_acc)
                    l1_lat = (float(mean_lat) if not np.isnan(mean_lat) else None)
                    l1_prauc = pr_auc(scores, yv)

            # L2
            l2_va_loss=None; l2_acc=None
            if ep <= L2_EPOCHS and dl2_va is not None:
                l2_model.eval(); va_loss=0; n=0; corr=0
                with torch.no_grad():
                    for xb,yb in dl2_va:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = l2_model(xb)
                        loss = F.cross_entropy(logits, yb)
                        va_loss += loss.item()*xb.size(0); n += xb.size(0)
                        pred = logits.argmax(-1); corr += (pred==yb).sum().item()
                if n>0:
                    l2_va_loss = va_loss/n; l2_acc = corr/n
                last_l2_val_acc = l2_acc

            # L3–L5
            def eval_acc_ce(dl, model):
                if dl is None: return (None, None)
                model.eval(); va_loss=0; n=0; corr=0
                with torch.no_grad():
                    for xb,yb in dl:
                        xb, yb = xb.to(device), yb.to(device)
                        # enc seq -> last
                        z_last = l35_enc(xb)[:,-1,:]
                        logits = model.head(z_last)
                        loss = F.cross_entropy(logits, yb)
                        va_loss += loss.item()*xb.size(0); n += xb.size(0)
                        corr += (logits.argmax(-1)==yb).sum().item()
                return (va_loss/n if n>0 else None, corr/n if n>0 else None)

            l3_va_loss,  l3_acc   = eval_acc_ce(dl3_va,   l3_model)  if ep<=L35_EPOCHS else (None,None)
            l4A_va_loss, l4A_acc  = eval_acc_ce(dl4A_va,  l4A_model) if ep<=L35_EPOCHS else (None,None)
            l4B_va_loss, l4B_acc  = eval_acc_ce(dl4B_va,  l4B_model) if ep<=L35_EPOCHS else (None,None)
            l5_01_va_loss, l5_01_acc = eval_acc_ce(dl5_01_va, l5_01) if ep<=L35_EPOCHS else (None,None)
            l5_23_va_loss, l5_23_acc = eval_acc_ce(dl5_23_va, l5_23) if ep<=L35_EPOCHS else (None,None)
            l5_45_va_loss, l5_45_acc = eval_acc_ce(dl5_45_va, l5_45) if ep<=L35_EPOCHS else (None,None)
            l5_67_va_loss, l5_67_acc = eval_acc_ce(dl5_67_va, l5_67) if ep<=L35_EPOCHS else (None,None)

            cascade_acc = None
            if ep<=L35_EPOCHS and X35_va_base is not None and len(X35_va_base)>0:
                with torch.no_grad():
                    xb = torch.from_numpy(X35_va_base).float().to(device)
                    z_last = l35_enc(xb)[:,-1,:]
                    tree_logp8 = compose_tree_logp_from_heads(
                        z_last, l3_model.head, l4A_model.head, l4B_model.head,
                        l5_01.head, l5_23.head, l5_45.head, l5_67.head
                    )
                    final_pred = tree_logp8.argmax(dim=1).cpu().numpy()
                    cascade_acc = float((final_pred == y35_va_base).mean())

        # ---------------- Hard Negative Mining (from EMA eval) ----------------
        # L2: X2_va_base / y2_va_base가 있을 때, 오분류 상위 K 추출
        if ep <= L2_EPOCHS and X2_va_base is not None and len(X2_va_base)>0:
            with torch.no_grad():
                xb = torch.from_numpy(X2_va_base).float().to(device)
                with (use_ema(ema) if ema else torch.no_grad()):
                    logits = l2_model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                wrong = (pred != y2_va_base)
                if wrong.any():
                    # margin 기반 가중
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    top = probs.max(axis=1)
                    margin = 1.0 - top
                    sel = np.argsort(-margin[wrong])[: min(1000, wrong.sum())]
                    Xbad = X2_va_base[wrong][sel]; ybad = y2_va_base[wrong][sel]
                    # append & cap
                    HN_L2_X = np.concatenate([HN_L2_X, Xbad], 0)
                    HN_L2_y = np.concatenate([HN_L2_y, ybad], 0)
                    if len(HN_L2_X) > HN_L2_MAX:
                        keep = np.random.choice(len(HN_L2_X), size=HN_L2_MAX, replace=False)
                        HN_L2_X = HN_L2_X[keep]; HN_L2_y = HN_L2_y[keep]

        # 8-way: X35_va_base / y35_va_base에서 오분류 상위 K
        if ep <= L35_EPOCHS and X35_va_base is not None and len(X35_va_base)>0:
            with torch.no_grad():
                xb = torch.from_numpy(X35_va_base).float().to(device)
                with (use_ema(ema) if ema else torch.no_grad()):
                    z_last = l35_enc(xb)[:,-1,:]
                    logits8 = motor8_head(z_last)
                pred = logits8.argmax(dim=1).cpu().numpy()
                wrong = (pred != y35_va_base)
                if wrong.any():
                    probs = F.softmax(logits8, dim=1).cpu().numpy()
                    top = probs.max(axis=1)
                    margin = 1.0 - top
                    sel = np.argsort(-margin[wrong])[: min(1000, wrong.sum())]
                    Xbad = X35_va_base[wrong][sel]; ybad = y35_va_base[wrong][sel]
                    HN_8_X = np.concatenate([HN_8_X, Xbad], 0)
                    HN_8_y = np.concatenate([HN_8_y, ybad], 0)
                    if len(HN_8_X) > HN_8_MAX:
                        keep = np.random.choice(len(HN_8_X), size=HN_8_MAX, replace=False)
                        HN_8_X = HN_8_X[keep]; HN_8_y = HN_8_y[keep]

        # ---------------------- LR Combo & LOG ----------------------
        val_combo = 0.0
        if 'l1_va_loss' in locals() and l1_va_loss is not None: val_combo += W_L1 * l1_va_loss
        if 'l2_va_loss' in locals() and l2_va_loss is not None: val_combo += W_L2 * l2_va_loss
        l35_vals = [v for v in ['l3_va_loss','l4A_va_loss','l4B_va_loss','l5_01_va_loss','l5_23_va_loss','l5_45_va_loss','l5_67_va_loss']
                    if locals().get(v, None) is not None]
        if len(l35_vals)>0:
            mean_l35 = np.mean([locals()[k] for k in l35_vals]).item()
            val_combo += W_L35 * mean_l35

        lr_groups = [pg['lr'] for pg in optimizer.param_groups]
        def fmt(x): return "nan" if x is None else f"{x:.4f}"
        print(f"\n[EPOCH {ep:03d}/{TOTAL_EPOCHS}] LR groups: {[f'{lr:.2e}' for lr in lr_groups]}")
        if tr_l1_loss is not None or ('l1_va_loss' in locals() and l1_va_loss is not None):
            print(f"  [L1] TrainLoss={tr_l1_loss if tr_l1_loss is not None else float('nan'):.4f} | "
                  f"ValLoss={l1_va_loss if 'l1_va_loss' in locals() and l1_va_loss is not None else float('nan'):.4f} | "
                  f"WinAcc={0.0 if 'l1_win_acc' not in locals() or l1_win_acc is None else l1_win_acc:.4f} | "
                  f"Prec={0.0 if 'l1_prec' not in locals() or l1_prec is None else l1_prec:.4f} | "
                  f"Rec={0.0 if 'l1_rec' not in locals() or l1_rec is None else l1_rec:.4f} | "
                  f"OnsetAcc={0.0 if 'l1_onset_acc' not in locals() or l1_onset_acc is None else l1_onset_acc:.4f} | "
                  f"PR-AUC={0.0 if 'l1_prauc' not in locals() or l1_prauc is None else l1_prauc:.4f} | "
                  f"Lat μ={l1_lat if 'l1_lat' in locals() and l1_lat is not None else float('nan'):.2f} | "
                  f"θ*={l1_theta_runtime:.2f}")

        if tr_l2_loss is not None or ('l2_va_loss' in locals() and l2_va_loss is not None):
            print(f"  [L2] TrainLoss={tr_l2_loss if tr_l2_loss is not None else float('nan'):.4f} | "
                  f"ValLoss={l2_va_loss if 'l2_va_loss' in locals() and l2_va_loss is not None else float('nan'):.4f} | "
                  f"Acc={0.0 if 'l2_acc' not in locals() or l2_acc is None else l2_acc:.4f}")

        print(f"  [L3]  VaLoss={fmt(locals().get('l3_va_loss'))}  Acc={fmt(locals().get('l3_acc'))}")
        print(f"  [L4A] VaLoss={fmt(locals().get('l4A_va_loss'))} Acc={fmt(locals().get('l4A_acc'))}   "
              f"[L4B] VaLoss={fmt(locals().get('l4B_va_loss'))} Acc={fmt(locals().get('l4B_acc'))}")
        print(f"  [L5_01] VaLoss={fmt(locals().get('l5_01_va_loss'))} Acc={fmt(locals().get('l5_01_acc'))}  "
              f"[L5_23] VaLoss={fmt(locals().get('l5_23_va_loss'))} Acc={fmt(locals().get('l5_23_acc'))}  "
              f"[L5_45] VaLoss={fmt(locals().get('l5_45_va_loss'))} Acc={fmt(locals().get('l5_45_acc'))}  "
              f"[L5_67] VaLoss={fmt(locals().get('l5_67_va_loss'))} Acc={fmt(locals().get('l5_67_acc'))}")
        print(f"  [Route] ok_rate={route_rate:.3f} (↑ 좋음), TF_onset≈{tf_linear_prob(ep, TOTAL_EPOCHS, TF_ONSET_START, TF_ONSET_END):.2f}, "
              f"TF_route≈{route_tfp_from_perf(ep, TOTAL_EPOCHS, last_l2_val_acc):.2f}, layout='{chosen_label_layout}'")
        print(f"  [Cascade Motor-ID] End-to-End Acc={fmt(locals().get('cascade_acc'))}  | Aux8/Cons Loss={fmt(tr_aux8)}/{fmt(tr_cons)}")

        # ---------------------- CHECKPOINT (EMA 파라미터 저장) -----------------
        if (ep == 1) or (ep % 10 == 0) or (ep == TOTAL_EPOCHS) or (val_combo < best_val_combo):
            best_val_combo = min(best_val_combo, val_combo)
            if ema: ema.copy_to()  # swap to EMA for saving
            save_obj = {
                "stage":"ALL",
                "meta": {"link_count": L, "motors_per_link": 8, "dt": DT_DEFAULT, "label_layout": chosen_label_layout},
                "znorm": {"mu": mu, "std": std} if USE_GLOBAL_ZNORM else None,
                "L1": {
                    "model_type": L1_MODEL_TYPE,
                    "state_dict": _to_cpu(l1_model.state_dict()),
                    "window": L1_WINDOW,
                    "threshold": l1_theta_runtime,
                    "val": {
                        "loss": float(locals().get('l1_va_loss')) if locals().get('l1_va_loss') is not None else None,
                        "win_acc": float(locals().get('l1_win_acc')) if locals().get('l1_win_acc') is not None else None,
                        "precision": float(locals().get('l1_prec')) if locals().get('l1_prec') is not None else None,
                        "recall": float(locals().get('l1_rec')) if locals().get('l1_rec') is not None else None,
                        "onset_acc": float(locals().get('l1_onset_acc')) if locals().get('l1_onset_acc') is not None else None,
                        "mean_latency": float(locals().get('l1_lat')) if locals().get('l1_lat') is not None else None,
                        "pr_auc": float(locals().get('l1_prauc')) if locals().get('l1_prauc') is not None else None
                    }
                },
                "L2": {
                    "temporal": True,
                    "state_dict": _to_cpu(l2_model.state_dict()),
                    "input_dim": int(L*L2_FEAT_DIM),
                    "window": L2_WINDOW,
                    "num_links": L,
                    "val": {
                        "loss": float(locals().get('l2_va_loss')) if locals().get('l2_va_loss') is not None else None,
                        "acc": float(locals().get('l2_acc')) if locals().get('l2_acc') is not None else None
                    }
                },
                "ENC_L35": {"state_dict": _to_cpu(l35_enc.state_dict()), "in_dim": 16, "window": L35_WIN},
                "L3":  {"state_dict": _to_cpu(l3_model.state_dict()),
                        "temporal": True,
                        "val": {"loss": float(locals().get('l3_va_loss')) if locals().get('l3_va_loss') is not None else None,
                                "acc": float(locals().get('l3_acc')) if locals().get('l3_acc') is not None else None}},
                "L4A": {"state_dict": _to_cpu(l4A_model.state_dict()),
                        "temporal": True,
                        "val": {"loss": float(locals().get('l4A_va_loss')) if locals().get('l4A_va_loss') is not None else None,
                                "acc": float(locals().get('l4A_acc')) if locals().get('l4A_acc') is not None else None}},
                "L4B": {"state_dict": _to_cpu(l4B_model.state_dict()),
                        "temporal": True,
                        "val": {"loss": float(locals().get('l4B_va_loss')) if locals().get('l4B_va_loss') is not None else None,
                                "acc": float(locals().get('l4B_acc')) if locals().get('l4B_acc') is not None else None}},
                "L5_01": {"state_dict": _to_cpu(l5_01.state_dict()),
                          "temporal": True,
                          "val": {"loss": float(locals().get('l5_01_va_loss')) if locals().get('l5_01_va_loss') is not None else None,
                                  "acc": float(locals().get('l5_01_acc')) if locals().get('l5_01_acc') is not None else None}},
                "L5_23": {"state_dict": _to_cpu(l5_23.state_dict()),
                          "temporal": True,
                          "val": {"loss": float(locals().get('l5_23_va_loss')) if locals().get('l5_23_va_loss') is not None else None,
                                  "acc": float(locals().get('l5_23_acc')) if locals().get('l5_23_acc') is not None else None}},
                "L5_45": {"state_dict": _to_cpu(l5_45.state_dict()),
                          "temporal": True,
                          "val": {"loss": float(locals().get('l5_45_va_loss')) if locals().get('l5_45_va_loss') is not None else None,
                                  "acc": float(locals().get('l5_45_acc')) if locals().get('l5_45_acc') is not None else None}},
                "L5_67": {"state_dict": _to_cpu(l5_67.state_dict()),
                          "temporal": True,
                          "val": {"loss": float(locals().get('l5_67_va_loss')) if locals().get('l5_67_va_loss') is not None else None,
                                  "acc": float(locals().get('l5_67_acc')) if locals().get('l5_67_acc') is not None else None}},
                "AUX8": {"state_dict": _to_cpu(motor8_head.state_dict()), "lambda": LAMBDA_AUX8, "lambda_cons": LAMBDA_CONS},
                "AGG": {
                    "val_combo": float(val_combo),
                    "cascade_acc": float(locals().get('cascade_acc')) if locals().get('cascade_acc') is not None else None,
                    "route_ok_rate": float(route_rate)
                },
                "HYPER": {
                    "W_L1": W_L1, "W_L2": W_L2, "W_L35": W_L35,
                    "L1": {"EPOCHS": L1_EPOCHS, "BS": L1_BS, "LR": L1_LR, "FOCAL": L1_FOCAL, "FOCAL_GAMMA": L1_FOCAL_GAMMA, "FOCAL_ALPHA": L1_FOCAL_ALPHA},
                    "L2": {"EPOCHS": L2_EPOCHS, "BS": L2_BS, "LR": L2_LR, "WINDOW": L2_WINDOW},
                    "L35": {"EPOCHS": L35_EPOCHS, "BS": L35_BS, "LR": L35_LR, "WINDOW": L35_WIN},
                    "TF": {"onset_start": TF_ONSET_START, "onset_end": TF_ONSET_END,
                           "warmup_l2": WARMUP_L2_EPOCHS, "warmup_l35": WARMUP_L35_EPOCHS,
                           "misroute_keep_p": MISROUTE_KEEP_P},
                    "STRATIFY": {
                        "force": FORCE_STRATIFY,
                        "L2_min_tr": L2_MIN_TR_PER_CLASS, "L2_min_va": L2_MIN_VA_PER_CLASS, "L2_max_tr": L2_MAX_TR_PER_CLASS,
                        "L35_min_tr": L35_MIN_TR_PER_CLASS, "L35_min_va": L35_MIN_VA_PER_CLASS, "L35_max_tr": L35_MAX_TR_PER_CLASS,
                        "L35_force_gt": L35_FORCE_GT_SEGMENTS, "L35_onset_jitter": L35_ONSET_JITTER
                    },
                    "AUG": {"time_mask_p": AUG_TIME_MASK_P, "feat_drop_p": AUG_FEAT_DROP_P, "feat_drop_frac": AUG_FEAT_DROP_FRAC},
                    "REG": {"tv": LAMBDA_TV, "label_smooth": CE_LABEL_SMOOTH},
                    "OPT": {"weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "ema": USE_EMA, "ema_decay": EMA_DECAY}
                }
            }
            torch.save(save_obj, COMBINED_CKPT_PATH)
            if ema: ema.restore()
            print(f"💾 Saved checkpoint → {COMBINED_CKPT_PATH}")

    print(f"✅ Done. Best combined val loss = {best_val_combo:.6f}")

# -------------------------- Z-Norm helper --------------------------
def compute_znorm_sampled(d_rel0, a_rel0, dt, max_seq=ZNORM_MAX_SEQ, max_samples=ZNORM_MAX_SAMPLES, seed=SEED):
    rng = np.random.RandomState(seed)
    S0 = d_rel0.shape[0]
    sel = rng.choice(S0, size=min(S0, max_seq), replace=False)
    per_link_tmp = build_state_features(d_rel0[sel], a_rel0[sel], dt=dt)
    x_flat = per_link_tmp.reshape(-1, per_link_tmp.shape[-1])
    if x_flat.shape[0] > max_samples:
        idx = rng.choice(x_flat.shape[0], size=max_samples, replace=False)
        x_flat = x_flat[idx]
    mu = x_flat.mean(axis=0).astype(np.float32)
    std = (x_flat.std(axis=0) + EPS_STD).astype(np.float32)
    del per_link_tmp, x_flat
    gc.collect()
    return mu, std

# -------------------------------------------------------------------
if __name__=="__main__":
    main()

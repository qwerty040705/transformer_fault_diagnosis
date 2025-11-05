import os, math, random, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler

# ============================================================
#                   Switches / Hyperparams
# ============================================================
SEED              = 107
EPOCHS            = 60
BATCH_SEQ         = 16
VAL_BS            = 16
FILES_PER_LOAD    = 10
DATA_ROOT         = "data_storage"

LSTM_HIDDEN       = 256
LSTM_LAYERS       = 2
LSTM_DROPOUT      = 0.30

LAMBDA_L1         = 1.0
LAMBDA_L2         = 0.5
LAMBDA_L3         = 0.3

LABEL_SMOOTH      = 0.00
USE_GLOBAL_ZNORM  = True

USE_PLATEAU_LR    = True
MIN_LR            = 1e-5

SAVE_DIR          = "LSTM_FDI_Hier_Atten"
CKPT_EVERY        = 20
SAVE_BEST         = True

EPS_STD           = 1e-6

# ============================================================
#      Basic SE(3) math utils
# ============================================================
def _vee_skew(A: np.ndarray) -> np.ndarray:
    return np.stack([A[...,2,1]-A[...,1,2],
                     A[...,0,2]-A[...,2,0],
                     A[...,1,0]-A[...,0,1]], axis=-1) / 2.0

def _so3_log(Rm: np.ndarray) -> np.ndarray:
    tr = np.clip((np.einsum('...ii', Rm)-1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    A = Rm - np.swapaxes(Rm, -1, -2)
    v = _vee_skew(A)
    sin_th = np.sin(theta)
    eps = 1e-9
    scale = np.where(np.abs(sin_th)[...,None]>eps, (theta/(sin_th+eps))[...,None], 1.0)
    w = v*scale
    return np.where((theta<1e-6)[...,None], v, w)

# ============================================================
#      External wrench observer
# ============================================================
def wrench_observer_discrete(alpha_k, alpha_km1, beta_k, Fext_km1, K1, K2, dt):
    return Fext_km1 + K2 @ (alpha_k - alpha_km1 - beta_k*dt) - (K1 @ Fext_km1)*dt

def batch_estimate_wrench_per_link(
    m_i, I_i, v_i, w_i, fu_i, tau_i, R_i,
    dt, K1_i, K2_i, g=np.array([0,0,-9.81], dtype=np.float64)
):
    T = v_i.shape[0]
    Fhat = np.zeros((T,6), dtype=np.float64)
    alpha_prev = np.concatenate([m_i*v_i[0], I_i @ w_i[0]], axis=0)
    Fprev = np.zeros(6, dtype=np.float64)
    if fu_i.ndim == 1:
        e3 = np.array([0,0,1], dtype=np.float64)
        fu_vec = (fu_i[:,None] * (R_i @ e3))
    else:
        fu_vec = fu_i
    for k in range(T):
        alpha_k = np.concatenate([m_i*v_i[k], I_i @ w_i[k]], axis=0)
        beta_k  = np.concatenate([fu_vec[k] - m_i*g, tau_i[k] - np.cross(w_i[k], I_i @ w_i[k])], axis=0)
        if k > 0:
            Fprev = wrench_observer_discrete(alpha_k, alpha_prev, beta_k, Fprev, K1_i, K2_i, dt)
        Fhat[k] = Fprev
        alpha_prev = alpha_k
    return Fhat

# ============================================================
#   Coupled 40D feature builder (causal, per-link)
#   = [F_{i-1}, F_i, F_{i+1}, (F_i-F_{i-1}), (F_i-F_{i+1}), dF/dt, u_i] (= 40)
#   🔒 GT 라벨 LATCH: 첫 고장 시점 이후 동일 모터 유지
# ============================================================
def build_coupled_features_causal(
    d_rel, a_rel, labels_motor, dt: float,
    masses=None, inertias=None, fu=None, tau=None, omega=None, vel=None, Rm=None
):
    """
    반환:
      X : (S, T, L, 40)
      y_l1 : (S, T)  = 0..L (0: no-fault, 1..L: link index+1)
      y_l2 : (S, T)  = 0..3 (region), no-fault은 -100(ignore)
      y_l3 : (S, T)  = 0..1 (pair idx), no-fault은 -100(ignore)
      mask : (S, T)  = 유효 프레임(=1). (패딩없으면 전부 1)
    """
    S, T, L = d_rel.shape[:3]

    if masses is None:   masses   = np.ones((L,), dtype=np.float64)
    if inertias is None: inertias = np.tile(np.eye(3, dtype=np.float64)[None,...], (L,1,1))
    if Rm is None:       Rm       = a_rel[..., :3, :3]

    # 속도/각속도 유도
    if vel is None or omega is None:
        p = a_rel[..., :3, 3]
        v_fd = np.zeros_like(p); v_fd[:,1:] = (p[:,1:] - p[:,:-1]) / max(dt, 1e-6)
        vel = v_fd
        rvec = _so3_log(Rm)
        w_fd = np.zeros_like(rvec); w_fd[:,1:] = (rvec[:,1:] - rvec[:,:-1]) / max(dt, 1e-6)
        omega = w_fd

    # 명령 입력 기본값
    if fu is None:  fu  = np.zeros((S,T,L))
    if tau is None: tau = np.zeros((S,T,L,3))

    # 관측기 게인
    K1 = np.eye(6)*0.05
    K2 = np.eye(6)*0.10

    # 출력 버퍼
    X = np.zeros((S, T, L, 40), dtype=np.float32)
    y_l1 = np.zeros((S, T), dtype=np.int64)
    y_l2 = np.full((S, T), -100, dtype=np.int64)   # ignore
    y_l3 = np.full((S, T), -100, dtype=np.int64)   # ignore
    mask = np.ones((S, T), dtype=np.float32)

    MOTORS_PER_LINK = 8

    for s in range(S):
        # per-link wrench 추정
        F_all = np.zeros((T, L, 6), dtype=np.float64)
        for i in range(L):
            m_i = float(masses[i]); I_i = inertias[i]
            v_i = vel[s,:,i,:]; w_i = omega[s,:,i,:]
            R_i = Rm[s,:,i,:,:]
            fu_i = fu[s,:,i] if fu.ndim==3 else fu[s,:,i,:]
            tau_i = tau[s,:,i,:]
            Fhat = batch_estimate_wrench_per_link(m_i, I_i, v_i, w_i, fu_i, tau_i, R_i, dt, K1, K2)
            F_all[:, i, :] = Fhat

        # dF/dt
        dF = np.zeros_like(F_all)
        dF[1:] = (F_all[1:] - F_all[:-1]) / max(dt, 1e-6)

        # u_i = [thrust, tau_xyz]
        if fu.ndim == 3:
            thrust = fu[s]              # (T,L)
            u4 = np.stack([thrust, tau[s,:,:,0], tau[s,:,:,1], tau[s,:,:,2]], axis=-1)  # (T,L,4)
        else:
            thrust = fu[s,:,:,2] if fu.shape[-1]==3 else fu[s,:,:,0]
            u4 = np.stack([thrust, tau[s,:,:,0], tau[s,:,:,1], tau[s,:,:,2]], axis=-1)

        # 라벨 생성
        lm = labels_motor[s]  # (T, 8L)
        idx = lm.argmax(axis=1)
        has_fault = (lm.max(axis=1) > 0.5)

        for t in range(T):
            for i in range(L):
                Fm1 = F_all[t, i-1, :] if i-1>=0 else np.zeros(6)
                Fi  = F_all[t, i,   :]
                Fp1 = F_all[t, i+1, :] if i+1<L  else np.zeros(6)
                dFi = dF[t, i, :]
                feat = np.concatenate([Fm1, Fi, Fp1, Fi-Fm1, Fi-Fp1, dFi, u4[t,i]], axis=0)  # 40
                X[s,t,i,:] = feat.astype(np.float32)

            if not has_fault[t]:
                y_l1[s,t] = 0
            else:
                m_global   = int(idx[t])
                link_idx   = m_global // MOTORS_PER_LINK      # 0..L-1
                motor_in_l = m_global % MOTORS_PER_LINK       # 0..7
                region_idx = motor_in_l // 2                  # 0..3
                pair_idx   = motor_in_l % 2                   # 0..1
                y_l1[s,t] = 1 + link_idx
                y_l2[s,t] = region_idx
                y_l3[s,t] = pair_idx

        # 🔒 GT 라벨 LATCH (처음 비-0 모터 이후 고정)
        latched = False
        kept_l1, kept_l2, kept_l3 = 0, -100, -100
        for t in range(T):
            if not latched and y_l1[s,t] != 0:
                latched = True
                kept_l1, kept_l2, kept_l3 = y_l1[s,t], y_l2[s,t], y_l3[s,t]
            if latched:
                y_l1[s,t] = kept_l1
                y_l2[s,t] = kept_l2
                y_l3[s,t] = kept_l3

    return X, y_l1, y_l2, y_l3, mask

# ============================================================
# collate (T, L, D) 시퀀스 패딩
# ============================================================
def collate_causal(batch):
    maxT = max(x.shape[0] for (x,_,_,_,_) in batch)
    L = batch[0][0].shape[1]; D = batch[0][0].shape[2]
    xs, y1s, y2s, y3s, ms = [], [], [], [], []
    for (x, y1, y2, y3, m) in batch:
        T = x.shape[0]
        xp  = torch.zeros(maxT, L, D)
        y1p = torch.full((maxT,), 0,    dtype=torch.long)
        y2p = torch.full((maxT,), -100, dtype=torch.long)
        y3p = torch.full((maxT,), -100, dtype=torch.long)
        mp  = torch.zeros(maxT)
        xp[:T]  = torch.from_numpy(x)
        y1p[:T] = torch.from_numpy(y1)
        y2p[:T] = torch.from_numpy(y2)
        y3p[:T] = torch.from_numpy(y3)
        mp[:T]  = torch.from_numpy(m)
        xs.append(xp); y1s.append(y1p); y2s.append(y2p); y3s.append(y3p); ms.append(mp)
    return torch.stack(xs,0), torch.stack(y1s,0), torch.stack(y2s,0), torch.stack(y3s,0), torch.stack(ms,0)

# ============================================================
# Temporal Attention (causal-window)
# ============================================================
class TemporalAttentionCausal(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, h_blth):
        # h_blth: (B, L, T, H)
        s = self.attn(h_blth).squeeze(-1)              # (B,L,T)
        s_max = s.max(dim=2, keepdim=True).values      # (B,L,1)
        exp_s = torch.exp(s - s_max)                   # (B,L,T)
        cum_exp = torch.cumsum(exp_s, dim=2)           # (B,L,T)
        weighted_h = exp_s.unsqueeze(-1) * h_blth      # (B,L,T,H)
        cum_weighted_h = torch.cumsum(weighted_h, dim=2)  # (B,L,T,H)
        ctx = cum_weighted_h / (cum_exp.unsqueeze(-1) + 1e-12)
        h_out = h_blth + ctx
        return h_out

# ============================================================
# Hierarchical LSTM Model
# ============================================================
class HierFDI(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, link_count, dropout=0.3):
        super().__init__()
        self.L = link_count
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.tattn = TemporalAttentionCausal(hidden_dim)

        # L1
        self.nofault_head = nn.Sequential(
            nn.Linear(hidden_dim*self.L, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.link_head    = nn.Linear(hidden_dim, 1)

        # L2
        self.region_head  = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 4)
        )

        # L3
        self.motor_head   = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 8)  # -> (4,2)
        )

    def forward(self, x_btld):
        B, T, L, D = x_btld.shape
        assert L == self.L
        x_bltd = x_btld.permute(0,2,1,3).contiguous()            # (B,L,T,D)
        x_flat = x_bltd.view(B*L, T, D)                          # (B*L,T,D)
        h, _ = self.lstm(x_flat)                                 # (B*L,T,H)
        h = self.drop(h)
        h_blth = h.view(B, L, T, self.hidden_dim)                # (B,L,T,H)

        # causal attention
        h_blth = self.tattn(h_blth)                              # (B,L,T,H)
        h_bt_l_h = h_blth.permute(0,2,1,3).contiguous()          # (B,T,L,H)

        # ----- L1 -----
        link_score = self.link_head(h_bt_l_h).squeeze(-1)        # (B,T,L)
        ctx = h_bt_l_h.reshape(B, T, L*self.hidden_dim)          # (B,T,LH)
        nof_score = self.nofault_head(ctx).squeeze(-1)           # (B,T)
        l1_logits = torch.cat([nof_score.unsqueeze(-1), link_score], dim=-1)  # (B,T,1+L)

        # ----- L2 -----
        reg_logits = self.region_head(h_bt_l_h)                  # (B,T,L,4)

        # ----- L3 -----
        mot_logits_8 = self.motor_head(h_bt_l_h)                 # (B,T,L,8)
        mot_logits = mot_logits_8.view(B, T, L, 4, 2)            # (B,T,L,4,2)
        return l1_logits, reg_logits, mot_logits

# ============================================================
# Losses
# ============================================================
def loss_level1(l1_logits, y1, mask, label_smooth=0.0):
    logits = l1_logits[mask>0.5]
    target = y1[mask>0.5]
    if logits.numel()==0: return l1_logits.new_zeros(())
    return F.cross_entropy(logits, target, label_smoothing=label_smooth)

def loss_level2(reg_logits, y1, y2, mask):
    valid = (mask>0.5) & (y1>0) & (y2>=0)
    if valid.sum()==0: return reg_logits.new_zeros(())
    b_idx, t_idx = torch.where(valid)
    link_idx = (y1[valid]-1)
    logits = reg_logits[b_idx, t_idx, link_idx]
    target = y2[valid]
    return F.cross_entropy(logits, target)

def loss_level3(mot_logits, y1, y2, y3, mask):
    valid = (mask>0.5) & (y1>0) & (y2>=0) & (y3>=0)
    if valid.sum()==0: return mot_logits.new_zeros(())
    b_idx, t_idx = torch.where(valid)
    link_idx  = (y1[valid]-1)
    region_idx= y2[valid]
    logits = mot_logits[b_idx, t_idx, link_idx, region_idx]  # (N,2)
    target = y3[valid]
    return F.cross_entropy(logits, target)

# ============================================================
# Latching (예측 logits에 계층 래치 적용)
# ============================================================
def apply_hier_latch_to_logits(l1_logits, reg_logits, mot_logits, mask):
    """
    계층 래치: 첫 고장 예측 시점 t0에서
      L1: 링크 c 고정
      L2: 해당 링크의 리전 r 고정
      L3: (링크 c, 리전 r)의 페어 p 고정
    이후 t>=t0 모든 유효 프레임에서 선택지 고정(나머지 -INF).
    """
    B, T, C = l1_logits.shape
    device = l1_logits.device

    l1_out  = l1_logits.clone()
    reg_out = reg_logits.clone()
    mot_out = mot_logits.clone()

    with torch.no_grad():
        l1_pred = l1_logits.argmax(-1)        # (B,T)
        valid   = (mask > 0.5)                # (B,T)

        for b in range(B):
            nz = torch.where((l1_pred[b] != 0) & valid[b])[0]
            if nz.numel() == 0:
                continue

            t0 = int(nz[0].item())
            c  = int(l1_pred[b, t0].item())   # 1..L (링크 인덱스는 c-1)

            # L2에서 t0 리전 r
            r_logits_t0 = reg_logits[b, t0, c-1]      # (4,)
            r = int(torch.argmax(r_logits_t0).item()) # 0..3

            # L3에서 t0 페어 p
            p_logits_t0 = mot_logits[b, t0, c-1, r]   # (2,)
            p = int(torch.argmax(p_logits_t0).item()) # 0..1

            # t>=t0 & valid
            time_mask = torch.zeros(T, dtype=torch.bool, device=device)
            time_mask[t0:] = True
            time_mask = time_mask & valid[b]
            if not time_mask.any():
                continue

            idx_t = torch.where(time_mask)[0]

            # ---- L1 고정
            cls_mask = torch.ones(C, dtype=torch.bool, device=device)
            cls_mask[c] = False
            for t in idx_t.tolist():
                l1_out[b, t, cls_mask] = -1e9

            # ---- L2 고정 (해당 링크만)
            for t in idx_t.tolist():
                bad_regions = torch.ones(4, dtype=torch.bool, device=device)
                bad_regions[r] = False
                reg_out[b, t, c-1, bad_regions] = -1e9

            # ---- L3 고정 (해당 링크·리전만)
            for t in idx_t.tolist():
                bad_pairs = torch.ones(2, dtype=torch.bool, device=device)
                bad_pairs[p] = False
                mot_out[b, t, c-1, r, bad_pairs] = -1e9

    return l1_out, reg_out, mot_out

@torch.no_grad()
def latch_level1(pred_l1_int, mask=None):
    """
    정수 예측 래치 (지표/출력 안전장치용)
    """
    B, T = pred_l1_int.shape
    for b in range(B):
        latched = False
        kept = 0
        for t in range(T):
            if mask is not None and mask[b, t] <= 0.5:
                continue
            if not latched and pred_l1_int[b, t] != 0:
                latched = True
                kept = int(pred_l1_int[b, t].item())
            if latched:
                pred_l1_int[b, t] = kept
    return pred_l1_int

# ============================================================
# Data Loading
# ============================================================
def load_10_npz_shards(link_count: int):
    data_dir = os.path.join(DATA_ROOT, f"link_{link_count}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    files = sorted([f for f in glob.glob(os.path.join(data_dir, "*.npz"))])
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files in {data_dir}")
    return files[:FILES_PER_LOAD]

def merge_npz(files):
    d_rel_list, a_rel_list, labels_list = [], [], []
    masses=None; inertias=None; fu=None; tau=None; omega=None; vel=None; Rm=None
    for path in files:
        dset = np.load(path, allow_pickle=True)
        d_rel_i  = dset["desired_link_rel"]
        a_rel_i  = dset["actual_link_rel"]
        labels_i = dset["label"]
        d_rel_list.append(d_rel_i); a_rel_list.append(a_rel_i); labels_list.append(labels_i)
        if "mass" in dset: masses = dset["mass"]
        if "inertia" in dset: inertias = dset["inertia"]
        if "cmd_force" in dset: fu = dset["cmd_force"]
        if "cmd_torque" in dset: tau = dset["cmd_torque"]
        if "omega" in dset: omega = dset["omega"]
        if "vel" in dset: vel = dset["vel"]
        if "R_body" in dset: Rm = dset["R_body"]
    d_rel = np.concatenate(d_rel_list, axis=0)
    a_rel = np.concatenate(a_rel_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return d_rel, a_rel, labels, masses, inertias, fu, tau, omega, vel, Rm

# ============================================================
# Main
# ============================================================
def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print("Using device:", device)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    try:
        link_count = int(input("How many links?: ").strip())
    except Exception:
        link_count = 1; print("Invalid input. Using link_count=1.")

    shard_paths = load_10_npz_shards(link_count)
    print(f"Loading {len(shard_paths)} shards...")
    d_rel, a_rel, labels_raw, masses, inertias, fu, tau, omega, vel, Rm = merge_npz(shard_paths)

    # label 반전 유지 (사용 환경 맞춤)
    labels_raw = 1.0 - labels_raw

    S, T, L = d_rel.shape[:3]
    MOTORS_PER_LINK = 8
    expected_motor_ch = L * MOTORS_PER_LINK

    # motor one-hot만 추출 (BG 채널 있으면 드랍)
    M_raw = labels_raw.shape[2]
    if M_raw == expected_motor_ch + 1:
        print("Detected BG channel in raw labels; dropping it (will derive no-fault from zero rows).")
        labels_motor = labels_raw[..., 1:].astype(np.float32)
    elif M_raw == expected_motor_ch:
        labels_motor = labels_raw.astype(np.float32)
    else:
        print(f"⚠️ Unexpected label size M_raw={M_raw}. Using last {expected_motor_ch} channels as motor labels.")
        labels_motor = labels_raw[..., -expected_motor_ch:].astype(np.float32)

    # ============================================================
    # 🔵 Feature 캐시: 첫 실행 때만 생성, 이후엔 즉시 로드
    # ============================================================
    cache_path = os.path.join(SAVE_DIR, f"cached_features_link{L}.npz")
    os.makedirs(SAVE_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        print(f"✅ Using cached features: {cache_path}")
        d = np.load(cache_path)
        X_np, y1_np, y2_np, y3_np, mask_np = d["X"], d["y1"], d["y2"], d["y3"], d["mask"]
    else:
        print("🧮 Building features (first run)...")
        dt = 0.01
        X_np, y1_np, y2_np, y3_np, mask_np = build_coupled_features_causal(
            d_rel, a_rel, labels_motor, dt,
            masses=masses, inertias=inertias, fu=fu, tau=tau, omega=omega, vel=vel, Rm=Rm
        )
        np.savez_compressed(cache_path, X=X_np, y1=y1_np, y2=y2_np, y3=y3_np, mask=mask_np)
        print(f"💾 Cached to {cache_path}")

    # Z-Norm (전역)
    X_all = torch.from_numpy(X_np).float()   # (S,T,L,D)
    y1 = torch.from_numpy(y1_np).long()
    y2 = torch.from_numpy(y2_np).long()
    y3 = torch.from_numpy(y3_np).long()
    m  = torch.from_numpy(mask_np).float()

    if USE_GLOBAL_ZNORM:
        mu  = X_all.reshape(-1, X_all.shape[-1]).mean(dim=0)     # D
        std = X_all.reshape(-1, X_all.shape[-1]).std(dim=0) + 1e-6
        X_all = (X_all - mu)/std
    else:
        mu = torch.zeros(X_all.shape[-1]); std = torch.ones(X_all.shape[-1])

    # train/val split (시퀀스 단위)
    n_tr = int(0.8*X_all.shape[0]); n_va = X_all.shape[0] - n_tr
    idx = torch.randperm(X_all.shape[0], generator=torch.Generator().manual_seed(SEED))
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]

    def ds_from_indices(idxs):
        return list(zip(
            X_all[idxs].numpy(),
            y1[idxs].numpy(),
            y2[idxs].numpy(),
            y3[idxs].numpy(),
            m[idxs].numpy()
        ))

    train_ds = ds_from_indices(tr_idx)
    val_ds   = ds_from_indices(va_idx)

    # ----- FG oversampling (시퀀스 단위 가중 샘플러) -----
    y1_tr = y1[tr_idx].numpy()
    fg_frac = (y1_tr > 0).mean(axis=1)                       # (n_tr,)
    weights = 1.0 + 3.0 * fg_frac
    weights = torch.from_numpy(weights.astype(np.float32))
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SEQ, sampler=sampler, shuffle=False,
                              collate_fn=collate_causal, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=VAL_BS,  shuffle=False,
                              collate_fn=collate_causal, num_workers=0, pin_memory=True)

    FEAT = X_all.shape[-1]
    model = HierFDI(FEAT, LSTM_HIDDEN, LSTM_LAYERS, link_count=L, dropout=LSTM_DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=MIN_LR
    ) if USE_PLATEAU_LR else None

    best_val=float('inf')
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path=os.path.join(SAVE_DIR, "best_LSTM_FDI_Hier.pth")
    last_path=os.path.join(SAVE_DIR, "last_LSTM_FDI_Hier.pth")

    for ep in range(1, EPOCHS+1):
        # ---------------- Train ----------------
        model.train(); tr_loss_sum=0; tr_steps=0
        for xb, y1b, y2b, y3b, mb in train_loader:
            xb, y1b, y2b, y3b, mb = xb.to(device), y1b.to(device), y2b.to(device), y3b.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)

            # 원시 예측
            l1_logits_raw, reg_logits_raw, mot_logits_raw = model(xb)

            # 🔒 계층 래치 (훈련 시에도 동일 적용)
            l1_logits, reg_logits, mot_logits = apply_hier_latch_to_logits(
                l1_logits_raw, reg_logits_raw, mot_logits_raw, mb
            )

            # 손실 (GT는 이미 latch됨)
            loss1 = loss_level1(l1_logits, y1b, mb, LABEL_SMOOTH)
            loss2 = loss_level2(reg_logits, y1b, y2b, mb)
            loss3 = loss_level3(mot_logits, y1b, y2b, y3b, mb)
            loss = LAMBDA_L1*loss1 + LAMBDA_L2*loss2 + LAMBDA_L3*loss3

            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss_sum += loss.item()*mb.sum().item(); tr_steps += mb.sum().item()
        train_loss = tr_loss_sum / max(tr_steps,1.0)

        # ---------------- Val ----------------
        model.eval(); val_loss_sum=0; val_steps=0
        last_batch=None
        with torch.no_grad():
            for xb, y1b, y2b, y3b, mb in val_loader:
                xb, y1b, y2b, y3b, mb = xb.to(device), y1b.to(device), y2b.to(device), y3b.to(device), mb.to(device)

                l1_logits_raw, reg_logits_raw, mot_logits_raw = model(xb)

                # 🔒 계층 래치 (검증/지표도 동일 규칙)
                l1_logits, reg_logits, mot_logits = apply_hier_latch_to_logits(
                    l1_logits_raw, reg_logits_raw, mot_logits_raw, mb
                )

                v1 = loss_level1(l1_logits, y1b, mb, LABEL_SMOOTH)
                v2 = loss_level2(reg_logits, y1b, y2b, mb)
                v3 = loss_level3(mot_logits, y1b, y2b, y3b, mb)
                vloss = LAMBDA_L1*v1 + LAMBDA_L2*v2 + LAMBDA_L3*v3
                val_loss_sum += vloss.item()*mb.sum().item(); val_steps += mb.sum().item()
                last_batch = (l1_logits, reg_logits, mot_logits, y1b, y2b, y3b, mb)

        val_loss = val_loss_sum / max(val_steps,1.0)

        if USE_PLATEAU_LR and scheduler is not None:
            scheduler.step(val_loss)

        # ---------------- Metrics 출력 ----------------
        if last_batch is not None:
            l1_logits, reg_logits, mot_logits, y1b, y2b, y3b, mb = last_batch

            # latched logits -> argmax
            l1_pred = l1_logits.argmax(-1)  # (B,T)
            l1_pred = latch_level1(l1_pred, mb)  # 안전장치(동일 결과)

            mask_valid = (mb > 0.5)
            acc = ((l1_pred == y1b) & mask_valid).sum().item() / max(mask_valid.sum().item(), 1)

            fg_mask = (y1b != 0) & mask_valid
            bg_mask = (y1b == 0) & mask_valid
            fg_correct = ((l1_pred == y1b) & fg_mask).sum().item()
            bg_correct = ((l1_pred == y1b) & bg_mask).sum().item()
            fg_total = fg_mask.sum().item(); bg_total = bg_mask.sum().item()
            fg_recall = fg_correct / max(fg_total, 1)
            bg_acc    = bg_correct / max(bg_total, 1)

            if fg_total>0:
                b_idx, t_idx = torch.where(fg_mask)
                link_idx = (y1b[fg_mask]-1)
                # reg/mot도 logits-latch가 적용되어 있으므로 시간 내내 고정됨
                reg_pred = reg_logits[b_idx, t_idx, link_idx].argmax(-1)
                reg_acc = (reg_pred == y2b[fg_mask]).sum().item() / max(fg_total,1)
                mot_pred = mot_logits[b_idx, t_idx, link_idx, y2b[fg_mask]].argmax(-1)
                mot_acc = (mot_pred == y3b[fg_mask]).sum().item() / max(fg_total,1)
            else:
                reg_acc = 0.0; mot_acc = 0.0

            # Onset latency / onset accuracy
            l1_pred_np = l1_pred.cpu().numpy()
            y1_np = y1b.cpu().numpy()
            mb_np = mb.cpu().numpy()
            latencies=[]; onset_hits=0; onset_total=0
            for b in range(l1_pred_np.shape[0]):
                valid = mb_np[b]>0.5
                t_true = np.where((y1_np[b]!=0) & valid)[0]
                t_pred = np.where((l1_pred_np[b]!=0) & valid)[0]
                if len(t_true)>0:
                    onset_total += 1
                    if len(t_pred)>0:
                        latencies.append(t_pred[0]-t_true[0])
                        if abs(t_pred[0]-t_true[0]) <= 5:
                            onset_hits += 1
            mean_lat = np.mean(latencies) if len(latencies)>0 else float('nan')
            fg_onset_acc = onset_hits / max(onset_total,1)

            conf_str = "HierV2(causal-attn) + Hierarchical logits-latch (train/val/print)"
        else:
            acc = fg_onset_acc = bg_acc = fg_recall = reg_acc = mot_acc = 0.0
            mean_lat = float('nan')
            conf_str = "None"

        lr_now = optimizer.param_groups[0]['lr']
        print(f"[{ep:03d}][LR {lr_now:.2e}] [Train] loss={train_loss:.4f} | [Val] loss={val_loss:.4f}")
        print(f"       [RealTime] Acc={acc:.4f} | FGrec={fg_recall:.4f} | BGacc={bg_acc:.4f} | RegAcc={reg_acc:.4f} | MotAcc={mot_acc:.4f} | OnsetAcc={fg_onset_acc:.4f} | Lat μ={mean_lat:.2f}")
        print(f"       [Confusions] {conf_str}")

        # 체크포인트
        if val_loss<best_val:
            best_val=val_loss
            torch.save({
                "model_state_dict":model.state_dict(),
                "train_mean":mu.cpu().numpy(),"train_std":std.cpu().numpy(),
                "input_dim":FEAT,"link_count":L,
                "feature_mode":"Coupled40D(causal+dF/dt)",
                "heads":"L1(L+1),L2(4),L3(2)"
            },best_path)
            print(f"✨ Saved best model (val={best_val:.4f})")

        torch.save(model.state_dict(), last_path)

if __name__=="__main__":
    main()

import os, math, random, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# ============================================================
#                   Switches / Hyperparams
# ============================================================
SEED              = 109
EPOCHS            = 80
BATCH_SEQ         = 16
VAL_BS            = 16
FILES_PER_LOAD    = 10
DATA_ROOT         = "data_storage"

LSTM_HIDDEN       = 256
LSTM_LAYERS       = 1
LSTM_DROPOUT      = 0.30

LAMBDA_M          = 7.0
LABEL_SMOOTH      = 0.01
USE_GLOBAL_ZNORM  = True
USE_COSINE_LR     = True
MIN_LR            = 1e-5
SAVE_DIR          = "LSTM_FDI_18L"
CKPT_EVERY        = 20
SAVE_BEST         = True

# ---- 18L 특징 관련 하이퍼파라미터 ----
PREFAULT_WIN      = 60
MIN_BASE_FRAMES   = 10
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
#      Cacace-style external wrench observer
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
#     18L Feature builder
# ============================================================
def _find_fault_time_from_labels(lbl_st: np.ndarray) -> int:
    argm = lbl_st.argmax(axis=1)
    idx = np.where(argm != 0)[0]
    return int(idx[0]) if idx.size > 0 else lbl_st.shape[0]

def build_18L_features_full(d_rel, a_rel, labels_motor, dt: float,
                             masses=None, inertias=None,
                             fu=None, tau=None, omega=None, vel=None, Rm=None):
    S, T, L = d_rel.shape[:3]
    if masses is None:
        masses = np.ones((L,), dtype=np.float64)
    if inertias is None:
        inertias = np.tile(np.eye(3, dtype=np.float64)[None,...], (L,1,1))
    if Rm is None:
        Rm = a_rel[..., :3, :3]
    if vel is None or omega is None:
        p = a_rel[..., :3, 3]
        v_fd = np.zeros_like(p); v_fd[:,1:] = (p[:,1:] - p[:,:-1]) / max(dt, 1e-6)
        vel = v_fd
        rvec = _so3_log(Rm)
        w_fd = np.zeros_like(rvec); w_fd[:,1:] = (rvec[:,1:] - rvec[:,:-1]) / max(dt, 1e-6)
        omega = w_fd
    if fu is None:  fu  = np.zeros((S,T,L))
    if tau is None: tau = np.zeros((S,T,L,3))
    K1 = np.eye(6) * 0.05
    K2 = np.eye(6) * 0.10

    X_list, Y_list = [], []
    for s in range(S):
        wrench_all = []
        for i in range(L):
            m_i = float(masses[i]); I_i = inertias[i]
            v_i = vel[s,:,i,:]; w_i = omega[s,:,i,:]
            R_i = Rm[s,:,i,:,:]
            fu_i = fu[s,:,i] if fu.ndim==3 else fu[s,:,i,:]
            tau_i = tau[s,:,i,:]
            Fhat = batch_estimate_wrench_per_link(m_i, I_i, v_i, w_i, fu_i, tau_i, R_i, dt, K1, K2)
            wrench_all.append(Fhat)
        W = np.concatenate([w for w in wrench_all], axis=1)
        t_f = _find_fault_time_from_labels(labels_motor[s])
        if t_f == 0:
            start, end = 0, min(T, PREFAULT_WIN)
        else:
            end, start = t_f, max(0, t_f - PREFAULT_WIN)
            if (end - start) < MIN_BASE_FRAMES:
                start = max(0, end - MIN_BASE_FRAMES)
        W0 = W[start:end, :] if end > start else W[0:min(T, PREFAULT_WIN), :]
        mu = W0.mean(axis=0); std = W0.std(axis=0) + EPS_STD
        Z = (W - mu[None, :]) / std[None, :]
        dW = np.zeros_like(W); dW[1:, :] = W[1:, :] - W[:-1, :]
        X_s = np.concatenate([W, Z, dW], axis=1).astype(np.float32)
        X_list.append(X_s[None, ...]); Y_list.append(labels_motor[s][None, ...].astype(np.float32))
    X = np.concatenate(X_list, axis=0); Y = np.concatenate(Y_list, axis=0)
    return X, Y

# ============================================================
# collate
# ============================================================
def collate_full_sequence(batch):
    maxL = max(x.shape[0] for (x,_) in batch)
    xs, ys, ms = [], [], []
    for (x, y) in batch:
        L, D, M = x.shape[0], x.shape[1], y.shape[1]
        xp = torch.zeros(maxL, D); yp = torch.zeros(maxL, M); mp = torch.zeros(maxL)
        xp[:L] = x; yp[:L] = y; mp[:L] = 1.0
        xs.append(xp); ys.append(yp); ms.append(mp)
    return torch.stack(xs,0), torch.stack(ys,0), torch.stack(ms,0)

# ============================================================
# LSTM Model
# ============================================================
class LSTM_FDI(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_motors, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.motor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_motors)
        )
    def forward(self, x_bt_d):
        y, _ = self.lstm(x_bt_d)
        y = self.dropout(y)
        mot = self.motor_head(y)
        return mot

# ============================================================
# Loss
# ============================================================
def loss_motor_ce(motor_logits, y_bin, mask, label_smooth=0.0):
    B, T, M = motor_logits.shape
    losses = []
    for b in range(B):
        valid = (mask[b] > 0.5)
        logits_b = motor_logits[b][valid]
        targets_b = y_bin[b][valid].argmax(dim=1)
        ce_b = F.cross_entropy(logits_b, targets_b, reduction='mean', label_smoothing=label_smooth)
        losses.append(ce_b)
    return torch.stack(losses).mean() if losses else motor_logits.new_zeros(())

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

    labels_raw = 1.0 - labels_raw

    S, T, L = d_rel.shape[:3]
    MOTORS_PER_LINK = 8
    expected_motor_ch = L * MOTORS_PER_LINK

    M_raw = labels_raw.shape[2]
    if M_raw == expected_motor_ch + 1:
        print("Detected BG channel in raw labels; dropping it and re-computing BG later.")
        labels_motor = labels_raw[..., 1:].astype(np.float32)
    elif M_raw == expected_motor_ch:
        labels_motor = labels_raw.astype(np.float32)
    else:
        print(f"⚠️ Unexpected label size M_raw={M_raw}. Using last {expected_motor_ch} channels as motor labels.")
        labels_motor = labels_raw[..., -expected_motor_ch:].astype(np.float32)

    labels_motor = (labels_motor > 0.5).astype(np.float32)
    multi_mask = (labels_motor.sum(axis=2) > 1)
    if np.any(multi_mask):
        n_multi = int(multi_mask.sum())
        print(f"⚠️ Cleaning {n_multi} frames with multiple motors active (keeping argmax).")
        idx_s, idx_t = np.where(multi_mask)
        for s, t in zip(idx_s, idx_t):
            k = int(labels_motor[s, t].argmax())
            labels_motor[s, t, :] = 0.0
            labels_motor[s, t, k] = 1.0

    M_phys = labels_motor.shape[2]
    print(f"Dataset: S={S}, T={T}, L={L}, M_motor={M_phys} (BG will be added later)")
    dt = 0.01

    X, labels_motor_used = build_18L_features_full(
        d_rel, a_rel, labels_motor, dt,
        masses=masses, inertias=inertias,
        fu=fu, tau=tau, omega=omega, vel=vel, Rm=Rm
    )

    any_fault = (labels_motor_used > 0.5).any(axis=2, keepdims=True).astype(np.float32)
    bg_ch = (1.0 - any_fault).astype(np.float32)
    y_bin_np = np.concatenate([bg_ch, labels_motor_used.astype(np.float32)], axis=2)

    multi_fault_frames = (y_bin_np[:, :, 1:].sum(axis=2) > 1).sum()
    if multi_fault_frames > 0:
        print(f"⚠️ Warning: {multi_fault_frames} frames have multi-motor labels; CE 단일 라벨 가정과 충돌 가능")

    print(f"Feature dim: {X.shape[2]} (18L = 6W + 6Z + 6ΔW)")

    X_all = torch.from_numpy(X).float()
    y_bin = torch.from_numpy(y_bin_np).float()
    n_tr = int(0.8*X_all.shape[0]); n_va = X_all.shape[0] - n_tr
    ds = TensorDataset(X_all, y_bin)
    train_ds, val_ds = random_split(ds, [n_tr, n_va], generator=torch.Generator().manual_seed(SEED))
    tr_idx = train_ds.indices; va_idx = val_ds.indices

    if USE_GLOBAL_ZNORM:
        mu  = X_all[tr_idx].reshape(-1, X_all.shape[2]).mean(dim=0)
        std = X_all[tr_idx].reshape(-1, X_all.shape[2]).std(dim=0) + 1e-6
        X_all = (X_all - mu)/std
        train_ds = TensorDataset(X_all[tr_idx], y_bin[tr_idx])
        val_ds   = TensorDataset(X_all[va_idx],  y_bin[va_idx])
    else:
        mu = torch.zeros(X_all.shape[2]); std = torch.ones(X_all.shape[2])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SEQ, shuffle=True, collate_fn=collate_full_sequence)
    val_loader   = DataLoader(val_ds, batch_size=VAL_BS, shuffle=False, collate_fn=collate_full_sequence)

    FEAT = X_all.shape[2]
    M_total = y_bin.shape[2]
    model = LSTM_FDI(FEAT, LSTM_HIDDEN, LSTM_LAYERS, M_total, LSTM_DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR) if USE_COSINE_LR else None

    best_val=float('inf')
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path=os.path.join(SAVE_DIR, "best_LSTM_FDI_18L.pth")
    last_path=os.path.join(SAVE_DIR, "last_LSTM_FDI_18L.pth")

    for ep in range(1, EPOCHS+1):
        model.train(); tr_loss_sum=0; tr_steps=0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            mot_logits = model(xb)
            loss = LAMBDA_M * loss_motor_ce(mot_logits, yb, mb, LABEL_SMOOTH)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss_sum += loss.item()*mb.sum().item(); tr_steps += mb.sum().item()
        train_loss = tr_loss_sum / max(tr_steps,1.0)

        model.eval(); val_loss_sum=0; val_steps=0; last_batch=None
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                mot_logits = model(xb)
                vloss = LAMBDA_M * loss_motor_ce(mot_logits, yb, mb, LABEL_SMOOTH)
                val_loss_sum += vloss.item()*mb.sum().item(); val_steps += mb.sum().item()
                last_batch=(mot_logits,yb,mb)
        val_loss = val_loss_sum / max(val_steps,1.0)

        # ---------------- Metrics ----------------
        if last_batch is not None:
            mot_logits, yb, mb = last_batch
            preds = mot_logits.argmax(2)
            target = yb.argmax(2)
            mask_valid = (mb > 0.5)

            # 전체 정확도
            acc = ((preds == target) & mask_valid).sum().item() / max(mask_valid.sum().item(), 1)

            # FG(고장) / BG(정상) 분리
            fg_mask = (target != 0) & mask_valid
            bg_mask = (target == 0) & mask_valid
            fg_correct = ((preds == target) & fg_mask).sum().item()
            bg_correct = ((preds == target) & bg_mask).sum().item()
            fg_total = fg_mask.sum().item()
            bg_total = bg_mask.sum().item()
            fg_recall = fg_correct / max(fg_total, 1)
            bg_acc = bg_correct / max(bg_total, 1)

            # FG precision (ID precision)
            fg_pred_mask = (preds != 0) & mask_valid
            fg_pred_correct = ((preds == target) & fg_pred_mask).sum().item()
            fg_pred_total = fg_pred_mask.sum().item()
            fg_id = fg_pred_correct / max(fg_pred_total, 1)

            # Onset latency / accuracy
            preds_np = preds.cpu().numpy()
            target_np = target.cpu().numpy()
            latencies = []
            onset_hits = 0
            onset_total = 0
            for b in range(preds_np.shape[0]):
                t_true = np.where(target_np[b] != 0)[0]
                t_pred = np.where(preds_np[b] != 0)[0]
                if len(t_true) > 0:
                    onset_total += 1
                    if len(t_pred) > 0:
                        latencies.append(t_pred[0] - t_true[0])
                        if abs(t_pred[0] - t_true[0]) <= 5:
                            onset_hits += 1
            mean_lat = np.mean(latencies) if len(latencies) > 0 else float('nan')
            fg_onset_acc = onset_hits / max(onset_total, 1)

            # Confusion report (상위 5개)
            confusions = {}
            for true_cls in range(1, M_total):
                for pred_cls in range(1, M_total):
                    if true_cls != pred_cls:
                        cnt = (((target == true_cls) & (preds == pred_cls)) & mask_valid).sum().item()
                        if cnt > 0:
                            confusions[f"T{true_cls}->P{pred_cls}"] = cnt
            conf_str = ", ".join([f"{k}:{v}" for k, v in sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:5]]) or "None"
        else:
            acc = fg_id = bg_acc = fg_recall = fg_onset_acc = 0.0
            mean_lat = float('nan')
            conf_str = "None"

        lr_now = optimizer.param_groups[0]['lr']
        print(f"[{ep:03d}][LR {lr_now:.2e}] [Train] loss={train_loss:.4f} | [Val] loss={val_loss:.4f}")
        print(f"       [RealTime] Acc={acc:.4f} | FG-ID={fg_id:.4f} | BGacc={bg_acc:.4f} | FGrec={fg_recall:.4f} | OnsetAcc={fg_onset_acc:.4f} | Lat μ={mean_lat:.2f}")
        print(f"       [Confusions] {conf_str}")

        if USE_COSINE_LR: scheduler.step()


        lr_now=optimizer.param_groups[0]['lr']
        print(f"[{ep:03d}][LR {lr_now:.2e}] [Train] loss={train_loss:.4f} | [Val] loss={val_loss:.4f}")

        if val_loss<best_val:
            best_val=val_loss
            torch.save({
                "model_state_dict":model.state_dict(),
                "train_mean":mu.cpu().numpy(),"train_std":std.cpu().numpy(),
                "input_dim":FEAT,"num_motors":M_total,"link_count":link_count,
                "feature_mode":"18L(W,Z,dW)"
            },best_path)
            print(f"✨ Saved best model (val={best_val:.4f})")

        torch.save(model.state_dict(), last_path)

# ============================================================
# ==== λ-Gating Latch 추가 (논문 방식: N=10, λ=0.6) ====
# ============================================================
def apply_latch_window(probs, N=10, lam=0.6):
    """
    논문식 지속 확신 게이팅: 최근 N프레임 동안 특정 모터 확률이 λ 이상일 때 고장 라치.
    probs : [T, M_total] (softmax 확률)
    return : preds_latched (T,) - 0=정상, 1~M=모터 index
    """
    T, M = probs.shape
    preds_latched = np.zeros(T, dtype=int)
    for t in range(N, T):
        win = probs[t - N:t]
        avg = np.mean(win, axis=0)
        cls = np.argmax(avg)
        conf = avg[cls]
        if cls != 0 and conf >= lam:
            preds_latched[t:] = cls
            break
    return preds_latched

def infer_with_latch(model, X_seq, mu, std, N=10, lam=0.6, device="cpu"):
    """
    학습된 모델로 추론 + λ-게이팅 적용
    """
    model.eval()
    x = torch.from_numpy((X_seq - mu) / std).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits[0], dim=-1).cpu().numpy()
    preds = apply_latch_window(probs, N=N, lam=lam)
    return preds, probs

if __name__=="__main__":
    main()

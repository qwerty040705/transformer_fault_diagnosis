# train_causal_tcn.py
import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

# ====================== Feature builders (원본 호환) ======================
def _vee_skew(A):
    return np.stack([A[...,2,1]-A[...,1,2],
                     A[...,0,2]-A[...,2,0],
                     A[...,1,0]-A[...,0,1]], axis=-1) / 2.0

def _so3_log(Rm):
    tr = np.clip((np.einsum('...ii', Rm) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    A = Rm - np.swapaxes(Rm, -1, -2)
    v = _vee_skew(A)
    sin_th = np.sin(theta); eps = 1e-9
    scale = np.where(np.abs(sin_th)[...,None] > eps, (theta/(sin_th+eps))[...,None], 1.0)
    w = v * scale
    return np.where((theta < 1e-6)[...,None], v, w)

def _rot_err_vec(R_des, R_act):
    R_rel = np.matmul(np.swapaxes(R_des, -1, -2), R_act)
    return _so3_log(R_rel)

def _rel_log_increment(R):  # (...,T,3,3) -> (...,T,3)
    Tdim = R.shape[-3]
    out = np.zeros(R.shape[:-2] + (3,), dtype=R.dtype)
    if Tdim > 1:
        R_prev = R[..., :-1, :, :]
        R_next = R[..., 1:, :, :]
        R_rel  = np.matmul(np.swapaxes(R_prev, -1, -2), R_next)
        out[..., 1:, :] = _so3_log(R_rel)
    return out

def _time_diff(x):
    d = np.zeros_like(x)
    if x.shape[-3] > 1:
        d[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
    return d

def _flatten_3x4(T):
    return T[..., :3, :4].reshape(*T.shape[:-2], 12)

def build_features_rel_only(d_rel, a_rel):
    """
    입력: d_rel,a_rel : (S,T,L,4,4)  (T_{i-1,i})
    출력: X: (S,T, 42*L) = [des12|act12|p_err|r_err|dp_des|dp_act|dr_des|dr_act] per-link concat
    """
    S, T, L = d_rel.shape[:3]
    des_12 = _flatten_3x4(d_rel)
    act_12 = _flatten_3x4(a_rel)
    p_des, R_des = d_rel[..., :3, 3], d_rel[..., :3, :3]
    p_act, R_act = a_rel[..., :3, 3], a_rel[..., :3, :3]
    p_err  = p_act - p_des
    r_err  = _rot_err_vec(R_des, R_act)
    dp_des = _time_diff(p_des)
    dp_act = _time_diff(p_act)

    R_des_SK = np.swapaxes(R_des, 1, 2)
    R_act_SK = np.swapaxes(R_act, 1, 2)
    dr_des_SK = _rel_log_increment(R_des_SK)
    dr_act_SK = _rel_log_increment(R_act_SK)
    dr_des = np.swapaxes(dr_des_SK, 1, 2)
    dr_act = np.swapaxes(dr_act_SK, 1, 2)

    feats = np.concatenate([des_12, act_12, p_err, r_err, dp_des, dp_act, dr_des, dr_act], axis=-1)
    S_, T_, L_, _ = feats.shape
    return feats.reshape(S_, T_, L_*42).astype(np.float32)

def build_features_legacy(desired, actual):
    """
    구포맷: desired/actual : (S,T,4,4) -> (S,T,42)
    """
    S, T = desired.shape[:2]
    des_12 = desired[:, :, :3, :4].reshape(S, T, 12)
    act_12 = actual[:,  :, :3, :4].reshape(S, T, 12)
    p_des, R_des = desired[..., :3, 3], desired[..., :3, :3]
    p_act, R_act = actual[...,  :3, 3], actual[...,  :3, :3]
    p_err  = p_act - p_des
    r_err  = _rot_err_vec(R_des, R_act)
    dp_des = _time_diff(p_des)
    dp_act = _time_diff(p_act)
    dr_des = _rel_log_increment(R_des)
    dr_act = _rel_log_increment(R_act)
    X = np.concatenate([des_12, act_12, p_err, r_err, dp_des, dp_act, dr_des, dr_act], axis=2).astype(np.float32)
    return X

# ====================== Causal TCN (인과 dilated 1D CNN) ======================
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        padding = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.remove = padding  # 오른쪽 패딩 제외 → 미래 미참조 보장
    def forward(self, x):      # x: [B, C, T]
        y = super().forward(x)
        if self.remove > 0:
            y = y[:, :, :-self.remove]
        return y

class TemporalBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k=3, d=1, drop=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(ch_in,  ch_out, k, dilation=d)
        self.conv2 = CausalConv1d(ch_out, ch_out, k, dilation=d)
        self.act   = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(drop)
        self.down  = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        if isinstance(self.down, nn.Conv1d):
            nn.init.kaiming_uniform_(self.down.weight, nonlinearity="linear")
    def forward(self, x):  # [B,C,T]
        y = self.act(self.conv1(x))
        y = self.drop(y)
        y = self.act(self.conv2(y))
        y = self.drop(y)
        return self.act(y + self.down(x))

class FaultDiagnosisTCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128, layers=6, k=3, dropout=0.1):
        super().__init__()
        # dilation: 1,2,4,8,16,32 → receptive field R ≈ 1 + (k-1)*(2^layers-1)
        chans = [input_dim] + [hidden]*layers
        dilations = [2**i for i in range(layers)]
        blocks = []
        for i in range(layers):
            blocks.append(TemporalBlock(chans[i], chans[i+1], k=k, d=dilations[i], drop=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, output_dim, 1)
        )
    def forward(self, x):           # x: [B, T, D]
        x = x.transpose(1, 2)       # -> [B, D, T]
        h = self.tcn(x)             # -> [B, H, T]
        y = self.head(h)            # -> [B, M, T]
        return y.transpose(1, 2)    # -> [B, T, M]

# ====================== Exact-All 특화 손실 (마스크 지원) ======================
def softmin_beta(x, beta, dim=-1):
    return -(1.0/beta) * torch.logsumexp(-beta * x, dim=dim)

def loss_exactall_and_only_masked(logits, yb, mask=None, beta=10.0, m0=1.5, gamma=2.0,
                                  lambda_and=1.0, lambda_fp=0.05, fp_margin=0.1):
    """
    logits, yb: (B,T,M), yb∈{0,1}; mask: (B,T) {1=유효, 0=pad}
    BCE 없이 Exact-All만 밀어줌 + 정상 프레임 FP 억제 항.
    """
    B, T, M = logits.shape
    if mask is None:
        mask = torch.ones(B, T, dtype=logits.dtype, device=logits.device)
    mask = mask.clamp(0, 1)

    y_pm = 2*yb - 1
    s = y_pm * logits                              # (B,T,M)

    p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
    q = torch.where(yb.bool(), p, 1.0 - p)         # (B,T,M)
    Q = q.prod(dim=2).detach()                     # (B,T) stop-grad
    w = (1.0 - Q).pow(gamma)                       # (B,T)

    softmin_s = softmin_beta(s, beta=beta, dim=2)  # (B,T)
    and_core = torch.relu(m0 - softmin_s)          # (B,T)
    L_and = ((w * and_core) * mask).sum() / (mask.sum() + 1e-9)

    # 정상 프레임(모든 모터 0) 중 유효 구간만
    is_normal = (yb.sum(dim=2) == 0).float() * mask
    if is_normal.sum() > 0:
        p_max = p.max(dim=2).values
        L_fp = torch.relu(p_max - fp_margin) * is_normal
        L_fp = L_fp.sum() / (is_normal.sum() + 1e-9)
    else:
        L_fp = logits.new_tensor(0.0)

    loss = lambda_and * L_and + lambda_fp * L_fp
    return loss, L_and.detach(), L_fp.detach()

# ====================== 누적+슬라이딩 윈도우 Collate ======================
def make_multi_windows(x: torch.Tensor, y: torch.Tensor, views_per_seq=3,
                       min_L=32, mode_mix=True, lookback_cap=None):
    """
    x,y: (T,D)/(T,M) 한 시퀀스.
    - 누적(prefix) 1~2개 + 슬라이딩 1개(기본) → 총 views_per_seq 개 반환
    """
    T = x.shape[0]
    assert T >= min_L
    windows = []
    for v in range(views_per_seq):
        if mode_mix and v < views_per_seq-1:
            # prefix: [0:L]
            L = random.randint(min_L, T)
            t0, t1 = 0, L
        else:
            # sliding: [t-L, t)
            t = random.randint(min_L, T)
            if lookback_cap is None:
                Lmax = t
            else:
                Lmax = min(t, lookback_cap)
            L = random.randint(min_L, max(min_L, Lmax))
            t0, t1 = t - L, t
        windows.append((x[t0:t1], y[t0:t1]))
    return windows  # list of (Tx,D)/(Tx,M)

def collate_multi_prefix(batch, views_per_seq=3, min_L=32, lookback_cap=None):
    # batch: list of (x:[T,D], y:[T,M])
    xs, ys, masks = [], [], []
    maxL = 0
    per_seq_windows = []
    for (x, y) in batch:
        wins = make_multi_windows(x, y, views_per_seq=views_per_seq,
                                  min_L=min_L, mode_mix=True, lookback_cap=lookback_cap)
        per_seq_windows.extend(wins)
        for (w, _) in wins:
            maxL = max(maxL, w.shape[0])
    # pad
    for (wX, wY) in per_seq_windows:
        L = wX.shape[0]
        D = wX.shape[1]; M = wY.shape[1]
        x_pad = torch.zeros(maxL, D, dtype=wX.dtype)
        y_pad = torch.zeros(maxL, M, dtype=wY.dtype)
        m_pad = torch.zeros(maxL, dtype=torch.float32)
        x_pad[:L] = wX; y_pad[:L] = wY; m_pad[:L] = 1.0
        xs.append(x_pad); ys.append(y_pad); masks.append(m_pad)
    X = torch.stack(xs, 0)   # [B', maxL, D]
    Y = torch.stack(ys, 0)   # [B', maxL, M]
    Mmask = torch.stack(masks, 0)  # [B', maxL]
    return X, Y, Mmask

# ====================== 간단 지표 (배치용) ======================
def batch_exact_all_at(probs, y_true, mask, th=0.5):
    pred = (probs >= th).to(torch.int32)
    yint = y_true.to(torch.int32)
    eq = (pred == yint).all(dim=2).float() * mask  # (B,T)
    return (eq.sum() / (mask.sum() + 1e-9)).item()

def batch_any_f1_at(probs, y_true, mask, th=0.5):
    pred = (probs >= th).to(torch.int32)
    yint = y_true.to(torch.int32)
    gt_any = (yint==1).any(dim=2)         # (B,T)
    pd_any = (pred==1).any(dim=2)
    tp = ((gt_any & pd_any).float() * mask).sum().item()
    fp = (((~gt_any) & pd_any).float() * mask).sum().item()
    fn = ((gt_any & (~pd_any)).float() * mask).sum().item()
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return (2*p*r/(p+r)) if (p+r)>0 else 0.0

# ============================ Train ===========================
if __name__ == "__main__":
    # ── 입력 받기 (링크 수) ──
    try:
        link_count = int(input("How many links?: ").strip())
    except Exception:
        link_count = 1
        print("[WARN] Invalid input. Fallback to link_count=1")

    # ── 장치 선택 및 프린트 ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("📥 device:", device)

    # ── 경로 탐색 ──
    here = os.path.dirname(os.path.abspath(__file__))
    candidate_roots = [
        os.path.dirname(os.path.dirname(here)),  # .../repo/Transformer/train_*.py 형태 대비
        os.path.dirname(here),                   # .../repo/train_*.py
        here
    ]
    data_path = None; repo_root = None
    for root in candidate_roots:
        p = os.path.join(root, f"data_storage/link_{link_count}/fault_dataset.npz")
        if os.path.exists(p):
            data_path = p; repo_root = root; break
    if data_path is None:
        # 기본 추정
        repo_root = os.path.dirname(os.path.dirname(here))
        data_path = os.path.join(repo_root, f"data_storage/link_{link_count}/fault_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Not found: {data_path}")

    # ── 하이퍼파라미터 ──
    batch_seq = 16          # DataLoader에 들어가는 '원 시퀀스' 개수
    views_per_seq = 3       # 한 시퀀스에서 뽑는 윈도우 수(누적x2 + 슬라이딩x1 느낌)
    batch_size = batch_seq  # collate 후에는 배치가 batch_seq*views_per_seq 로 늘어남
    epochs = 200
    lr, wd, seed = 1e-3, 1e-4, 42
    min_L = 32
    lookback_cap = 512      # 슬라이딩 윈도우 최대 길이(온라인 lookback 감각)
    LAMBDA_FP = 0.05

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # ── 데이터 로드 (스키마 자동 감지) ──
    d = np.load(data_path, allow_pickle=True)
    keys = set(d.files)
    if {"desired_link_rel","actual_link_rel","desired_link_cum","actual_link_cum","label"}.issubset(keys):
        d_rel = d["desired_link_rel"]   # (S,T,L,4,4)
        a_rel = d["actual_link_rel"]
        labels = d["label"]             # (S,T,M=8L)
        dt = float(d.get("dt", 0.01))
        S, T, L = d_rel.shape[:3]
        M = labels.shape[2]; assert M == 8*L
        X = build_features_rel_only(d_rel, a_rel)  # (S,T, 42*L)
        y = (1.0 - labels).astype(np.float32)      # 1=fault, 0=normal
        print(f"Loaded REL-ONLY  S={S}, T={T}, L={L}, M={M}, FEAT={X.shape[2]} | epochs={epochs}")
    elif {"desired","actual","label"}.issubset(keys):
        desired = d["desired"]; actual = d["actual"]; labels = d["label"]
        dt = float(d.get("dt", 0.01))
        S, T = desired.shape[:2]; M = labels.shape[2]
        X = build_features_legacy(desired, actual) # (S,T,42)
        y = (1.0 - labels).astype(np.float32)
        print(f"Loaded LEGACY     S={S}, T={T}, M={M}, FEAT={X.shape[2]} | epochs={epochs}")
    else:
        raise KeyError(f"Unsupported .npz schema. keys={sorted(keys)}")
    FEAT_DIM = X.shape[2]

    # ── Dataset & split ──
    full_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_sz = int(0.8 * S); val_sz = S - train_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz], generator=torch.Generator().manual_seed(seed))

    # ── 표준화(Train split 기준) ──
    X_train = train_ds.dataset.tensors[0][train_ds.indices]  # (train_S, T, FEAT_DIM)
    mu = X_train.reshape(-1, FEAT_DIM).mean(0)
    std = X_train.reshape(-1, FEAT_DIM).std(0) + 1e-6
    def norm_tensor(a: torch.Tensor): return (a - mu) / std
    X_all = full_ds.tensors[0]; y_all = full_ds.tensors[1]
    X_norm = norm_tensor(X_all)
    dataset_all = TensorDataset(X_norm, y_all)
    train_ds, val_ds = random_split(dataset_all, [train_sz, val_sz], generator=torch.Generator().manual_seed(seed))

    # ── DataLoader: 누적+슬라이딩 윈도우 확장 collate ──
    def _train_collate(batch):
        # batch: list of (x:[T,D], y:[T,M]) 원 시퀀스들
        return collate_multi_prefix(batch, views_per_seq=views_per_seq, min_L=min_L, lookback_cap=lookback_cap)
    train_loader = DataLoader(train_ds, batch_size=batch_seq, shuffle=True, drop_last=False, collate_fn=_train_collate)

    def _val_collate(batch):
        # 검증은 전체 길이(T) 사용. mask=1.
        xs, ys, ms = [], [], []
        for (x, y) in batch:
            xs.append(x); ys.append(y); ms.append(torch.ones(x.shape[0], dtype=torch.float32))
        Xb = torch.stack(xs,0); Yb = torch.stack(ys,0); Mb = torch.stack(ms,0)
        return Xb, Yb, Mb
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False, collate_fn=_val_collate)

    # ── 모델/옵티마이저/스케줄러 ──
    model = FaultDiagnosisTCN(input_dim=FEAT_DIM, output_dim=M, hidden=128, layers=6, k=3, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    # ── 체크포인트 경로 ──
    ckpt_dir = os.path.join(repo_root, "TCN"); os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"TCN_link_{link_count}_RELonly_CAUSAL.pth")

    # ── 학습 루프 ──
    for ep in range(1, epochs+1):
        beta = 4.0 + 8.0 * min(1.0, ep/30.0)   # 4→12
        m0   = 0.5 + 1.0 * min(1.0, ep/30.0)   # 0.5→1.5

        model.train(); tr_sum = 0.0; tr_count = 0
        # 간단히: 첫 미니배치에서 훈련 지표 한번 찍기
        printed_batch_metrics = False

        for (xb, yb, mb) in train_loader:  # xb:[B',L,D] yb:[B',L,M] mb:[B',L]
            xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
            opt.zero_grad()
            logits = model(xb)                        # (B',L,M)
            loss, L_and, L_fp = loss_exactall_and_only_masked(
                logits, yb, mask=mb,
                beta=beta, m0=m0, gamma=2.0,
                lambda_and=1.0, lambda_fp=LAMBDA_FP, fp_margin=0.1
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_sum += loss.item() * mb.sum().item()
            tr_count += mb.sum().item()

            if not printed_batch_metrics:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    exact_b = batch_exact_all_at(probs, yb, mb, th=0.5)
                    anyf1_b = batch_any_f1_at(probs, yb, mb, th=0.5)
                    pos_rate = (probs>=0.5).float().mean().item()
                print(f"[ep{ep:03d} train-batch] loss={loss.item():.4f} | ExactAll@0.5={exact_b:.4f} | AnyF1@0.5={anyf1_b:.4f} | pos@0.5={pos_rate:.4f}")
                printed_batch_metrics = True

        tr_loss = tr_sum / max(tr_count, 1.0)

        # ── Validation ──
        model.eval(); val_sum = 0.0; val_count = 0
        probs_col, trues_col = [], []
        with torch.no_grad():
            for (xb, yb, mb) in val_loader:
                xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
                logits = model(xb)
                loss, L_and, L_fp = loss_exactall_and_only_masked(
                    logits, yb, mask=mb,
                    beta=beta, m0=m0, gamma=2.0,
                    lambda_and=1.0, lambda_fp=LAMBDA_FP, fp_margin=0.1
                )
                val_sum += loss.item() * mb.sum().item()
                val_count += mb.sum().item()
                p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
                probs_col.append(p.cpu()); trues_col.append(yb.cpu())
        val_loss = val_sum / max(val_count, 1.0)
        sched.step(val_loss)

        # ── 지표 리포트 ──
        val_probs = torch.cat(probs_col, 0)  # (N,T,M)
        val_true  = torch.cat(trues_col, 0)  # (N,T,M)
        ones_mask = torch.ones(val_probs.shape[:2], dtype=torch.float32)  # (N,T)
        exact_at05 = batch_exact_all_at(val_probs, val_true, ones_mask, th=0.5)
        anyf1_at05 = batch_any_f1_at(val_probs, val_true, ones_mask, th=0.5)
        pos_rate = (val_probs>=0.5).float().mean().item()

        print(f"[{ep:03d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} "
              f"| ExactAll@0.5={exact_at05:.4f} | AnyFaultF1@0.5={anyf1_at05:.4f} | pos@0.5={pos_rate:.4f}")

    # ── Save ckpt ──
    torch.save({
        "model_state": model.state_dict(),
        "train_mean": mu.cpu().numpy(), "train_std": std.cpu().numpy(),
        "input_dim": FEAT_DIM, "T": T, "M": M,
        "cfg": dict(model="TCN", hidden=128, layers=6, k=3, dropout=0.1,
                    training="multi-prefix + sliding (causal)"),
        "label_convention": "1=fault, 0=normal",
        "loss": "Exact-All AND-softmin + FP hinge (no BCE), masked",
        "loss_params": dict(m0_schedule="0.5->1.5@30ep", gamma=2.0, beta_schedule="4->12@30ep",
                            fp_margin=0.1, lambda_and=1.0, lambda_fp=0.05),
        "features": "REL-only/LEGACY 지원, per-link 42D × L concat",
        "dt": float(dt),
    }, save_path)
    print("✅ saved:", save_path)

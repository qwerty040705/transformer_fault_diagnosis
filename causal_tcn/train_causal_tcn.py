# ------------------------------------------------------------
# Causal fault diagnosis with latch:
# - Per-frame MOTOR logits (always pick 1 motor as candidate, no BG class)
# - Per-frame GATE logit (fault-present probability)
# - Inference: once GATE triggers, latch to a motor and never return to BG (absorbing)
# - Loss = Weighted Gate-BCE + Anti-reversion (post-onset) + Motor Focal+Margin (on positive frames)
# - Metrics (validation): Top1Acc RAW / BGacc / FGrec computed on LATCHED predictions
# - Twist-only features (SE(3) log-rot + trans, + deltas), NO 12D flatten
# ------------------------------------------------------------
import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

# ======================= Switches / Hyperparams =======================
MODEL_TYPE     = "tcn"      # "tcn" | "transformer"
WIDER_MODEL    = False
SEED           = 1
EPOCHS         = 120
BATCH_SEQ      = 16
VAL_BS         = 16
VIEWS_PER_SEQ  = 3
MIN_WIN        = 32
LOOKBACK_CAP   = 512
POS_BIAS_P     = 0.9      # prefer windows that include faults

# Gate (fault-present) loss weights
GATE_POS_W     = 1.5      # weight on positive frames (labels any-fault=1)
GATE_NEG_W     = 2.0      # weight on background frames (labels any-fault=0) -> suppress FP
LAMBDA_GATE    = 1.0

# Anti-reversion (post-onset monotonicity) penalty
LAMBDA_REV     = 0.8

# Motor identification (only on positive frames)
LAMBDA_MOTOR   = 1.0
MOTOR_GAMMA    = 2.5
MOTOR_MARGIN   = 0.20     # AM-Softmax style margin
MOTOR_SCALE    = 10.0
MOTOR_ALPHA    = 1.0      # overall weight inside focal term (프레임 가중이므로 1.0 권장)

# Real-time latch thresholds (validation-time inference)
LATCH_THETA    = 0.65     # threshold on gate probability to latch
LATCH_KN_NW    = 5        # require >=K hits in the last N to latch (stability)
LATCH_KN_K     = 3
LATCH_MOTOR_VOTE_N = 5    # motor selection at latch: majority over last N frames (argmax of logits)

# AMP
USE_AMP        = True

# ========================= Feature utilities =========================
def _vee_skew(A: np.ndarray) -> np.ndarray:
    return np.stack([A[...,2,1]-A[...,1,2], A[...,0,2]-A[...,2,0], A[...,1,0]-A[...,0,1]], axis=-1)/2.0

def _so3_log(Rm: np.ndarray) -> np.ndarray:
    tr = np.clip((np.einsum('...ii', Rm)-1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    A = Rm - np.swapaxes(Rm, -1, -2)
    v = _vee_skew(A)
    sin_th = np.sin(theta); eps=1e-9
    scale = np.where(np.abs(sin_th)[...,None]>eps, (theta/(sin_th+eps))[...,None], 1.0)
    w = v*scale
    return np.where((theta<1e-6)[...,None], v, w)

def _time_diff(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x)
    if x.shape[-3] > 1: d[...,1:,:] = x[...,1:,:] - x[..., :-1,:]
    return d

def _twist_from_T(T: np.ndarray) -> np.ndarray:
    Rm = T[..., :3, :3]
    t  = T[..., :3, 3]
    rvec = _so3_log(Rm)
    return np.concatenate([rvec, t], axis=-1)  # (...,6)

def build_features_twist_only(d_rel: np.ndarray, a_rel: np.ndarray) -> np.ndarray:
    S, T, L = d_rel.shape[:3]
    tw_des  = _twist_from_T(d_rel)      # (S,T,L,6)
    tw_act  = _twist_from_T(a_rel)      # (S,T,L,6)
    tw_err  = tw_act - tw_des           # (S,T,L,6)
    d_des   = _time_diff(tw_des)        # (S,T,L,6)
    d_act   = _time_diff(tw_act)        # (S,T,L,6)
    feats   = np.concatenate([tw_des, tw_act, tw_err, d_des, d_act], axis=-1)  # (S,T,L,30)
    return feats.reshape(S, T, L*30).astype(np.float32)

# ========================= Windowing / Collate ========================
def _make_windows_pos_biased(x: torch.Tensor, y_bin: torch.Tensor,
                             views_per_seq=3, min_L=32, lookback_cap=512, pos_bias_p=0.9):
    """
    x: [T,D], y_bin: [T,M] (1=fault, 0=normal) per motor; assume <=1 positive per frame.
    Returns list of (x_window, y_window).
    """
    T = x.shape[0]; assert T >= min_L
    pos_t = (y_bin.max(dim=1).values > 0.5).nonzero(as_tuple=False).squeeze(-1)
    wins = []
    for _ in range(views_per_seq):
        use_pos = (torch.rand(1).item() < pos_bias_p) and (pos_t.numel() > 0)
        if use_pos:
            t = int(pos_t[torch.randint(0, pos_t.numel(), (1,)).item()])
        else:
            t = random.randint(min_L, T)
        Lmax = t if lookback_cap is None else min(t, lookback_cap)
        L = random.randint(min_L, max(min_L, Lmax))
        t0, t1 = max(0, t-L), t
        wins.append((x[t0:t1], y_bin[t0:t1]))
    return wins

def collate_multi_prefix(batch, views_per_seq=3, min_L=32, lookback_cap=512, pos_bias_p=0.9):
    xs, ys, ms = [], [], []
    all_wins = []; maxL = 0
    for (x_seq, y_seq) in batch:
        ws = _make_windows_pos_biased(x_seq, y_seq, views_per_seq, min_L, lookback_cap, pos_bias_p)
        all_wins.extend(ws)
        for (xw, _) in ws: maxL = max(maxL, xw.shape[0])
    # ensure >=30% positive windows
    is_pos = [w[1].max().item() > 0.5 for w in all_wins]
    need = int(0.3*len(all_wins)) - sum(is_pos)
    while need > 0 and len(batch) > 0:
        x_seq, y_seq = random.choice(batch)
        extra = _make_windows_pos_biased(x_seq, y_seq, views_per_seq=1, min_L=min_L, lookback_cap=lookback_cap, pos_bias_p=1.0)
        all_wins.extend(extra); need -= 1
        for (xw, _) in extra: maxL = max(maxL, xw.shape[0])
    for (xw, yw) in all_wins:
        L, D, M = xw.shape[0], xw.shape[1], yw.shape[1]
        xp = torch.zeros(maxL, D); yp = torch.zeros(maxL, M); mp = torch.zeros(maxL)
        xp[:L] = xw; yp[:L] = yw; mp[:L] = 1.0
        xs.append(xp); ys.append(yp); ms.append(mp)
    return torch.stack(xs,0), torch.stack(ys,0), torch.stack(ms,0)

def collate_full_sequence(batch):
    maxL = max(x.shape[0] for (x,_) in batch)
    xs, ys, ms = [], [], []
    for (x, y) in batch:
        L, D, M = x.shape[0], x.shape[1], y.shape[1]
        xp = torch.zeros(maxL, D); yp = torch.zeros(maxL, M); mp = torch.zeros(maxL)
        xp[:L] = x; yp[:L] = y; mp[:L] = 1.0
        xs.append(xp); ys.append(yp); ms.append(mp)
    return torch.stack(xs,0), torch.stack(ys,0), torch.stack(ms,0)

# =================== Label utilities ====================
def ybin_to_class(y_bin: torch.Tensor) -> torch.Tensor:
    """
    y_bin: [B,T,M] (1=fault) -> y_cls: [B,T] in {0..M}; 0=normal, i>0= motor i
    """
    is_bg = (y_bin.sum(dim=2) == 0)
    idx = y_bin.argmax(dim=2) + 1
    return idx.masked_fill(is_bg, 0)

# ============================= Models ================================
class PositionalEncodingBF(nn.Module):
    def __init__(self, d_model:int, max_len:int=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div); pe[:,1::2] = torch.cos(position*div)
        self.register_buffer("pe", pe)
    def forward(self, x):  # [B,T,d]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class FaultTransformerGateMotor(nn.Module):
    def __init__(self, input_dim:int, motors:int, d_model=128, heads=8, layers=4, ff=256, drop=0.1, causal=True):
        super().__init__()
        self.causal = causal
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncodingBF(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads,
                                               dim_feedforward=ff, dropout=drop, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        if torch.backends.mps.is_available():
            try: self.enc.use_nested_tensor = False
            except Exception: pass
        self.drop = nn.Dropout(drop)
        self.gate_head  = nn.Linear(d_model, 1)      # logits -> [B,T,1]
        self.motor_head = nn.Linear(d_model, motors) # logits -> [B,T,M]
        self.d_model = d_model

    def forward(self, x, pad_mask=None):
        # x:[B,T,D], pad_mask:[B,T](1=valid)
        B,T,_ = x.shape
        z = self.in_proj(x) * math.sqrt(self.d_model)
        z = self.pos(z); z = self.drop(z)
        key_padding_mask = (pad_mask==0) if pad_mask is not None else None
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(torch.ones(T,T,dtype=torch.bool, device=x.device), diagonal=1)
        h = self.enc(z, mask=attn_mask, src_key_padding_mask=key_padding_mask)  # [B,T,d]
        gate_logits  = self.gate_head(h).squeeze(-1)   # [B,T]
        motor_logits = self.motor_head(h)              # [B,T,M]
        return gate_logits, motor_logits

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        padding = (k-1)*dilation
        super().__init__(in_ch, out_ch, k, padding=padding, dilation=dilation)
        self.remove = padding
    def forward(self, x):
        y = super().forward(x)
        return y[:,:, :-self.remove] if self.remove>0 else y

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1, p=0.1):
        super().__init__()
        self.c1 = CausalConv1d(in_ch, out_ch, k, dilation)
        self.c2 = CausalConv1d(out_ch, out_ch, k, dilation)
        self.relu = nn.ReLU(inplace=True); self.drop = nn.Dropout(p)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
        nn.init.kaiming_uniform_(self.c1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.c2.weight, nonlinearity="relu")
        if isinstance(self.down, nn.Conv1d):
            nn.init.kaiming_uniform_(self.down.weight, nonlinearity="linear")
    def forward(self, x):
        out = self.drop(self.relu(self.c1(x)))
        out = self.drop(self.relu(self.c2(out)))
        return self.relu(out + self.down(x))

class FaultTCNGateMotor(nn.Module):
    def __init__(self, input_dim, motors, hidden=128, layers=6, k=3, p=0.1):
        super().__init__()
        blocks = []; in_ch = input_dim
        for i in range(layers):
            d = 2**i
            blocks.append(TemporalBlock(in_ch, hidden, k, dilation=d, p=p))
            in_ch = hidden
        self.tcn  = nn.Sequential(*blocks)
        self.gate_head  = nn.Conv1d(hidden, 1, 1)     # [B,1,T]
        self.motor_head = nn.Conv1d(hidden, motors, 1) # [B,M,T]

    def forward(self, x, pad_mask=None):  # x:[B,T,D]
        h = self.tcn(x.transpose(1,2))                 # [B,H,T]
        gate_logits  = self.gate_head(h).squeeze(1).transpose(0,0) # [B,T]
        motor_logits = self.motor_head(h).transpose(1,2)           # [B,T,M]
        return gate_logits, motor_logits

# ============================ Losses ================================
def gate_weighted_bce_with_logits(logits, y_any, w_pos=1.5, w_neg=2.0, mask=None):
    """
    logits: [B,T], y_any: [B,T] in {0,1}, mask: [B,T]
    Returns weighted BCE averaged over valid steps.
    """
    if mask is None:
        mask = torch.ones_like(y_any)
    pos = (y_any > 0.5).float()
    neg = 1.0 - pos
    # standard BCE with logits per element
    bce = F.binary_cross_entropy_with_logits(logits, y_any, reduction='none')  # [B,T]
    w = (w_pos * pos + w_neg * neg) * mask
    loss = (bce * w).sum() / (mask.sum() + 1e-9)
    return loss

def gate_antireversion_penalty(logits, y_any, mask=None, lam=0.8):
    """
    Penalize gate decrease after fault has happened (monotonic encouragement).
    logits: [B,T] (pre-sigmoid), y_any: [B,T] (0/1), mask: [B,T]
    """
    if mask is None:
        mask = torch.ones_like(y_any)
    g = torch.sigmoid(logits)  # [B,T]
    # cumulative label: once fault appears, it stays 1
    y_cum = torch.cumsum((y_any > 0.5).float(), dim=1).clamp(max=1.0)  # [B,T]
    # regions where we want monotonic non-decreasing gate
    m = (y_cum > 0.5).float() * mask
    # penalty on negative diffs: g_{t-1} - g_t > 0
    diff = F.pad(g[:,1:] - g[:,:-1], (1,0), value=0.0)  # [B,T], first diff=0
    pen = torch.relu(-diff) * m
    return lam * (pen.sum() / (m.sum() + 1e-9))

def motor_focal_margin_loss_framewise(motor_logits, y_cls, mask=None,
                                      gamma=2.5, margin=0.2, scale=10.0, alpha=1.0):
    """
    motor_logits: [B,T,M], y_cls: [B,T] in {0..M}; loss only where y_cls>0 (positive frames)
    Returns average focal+margin loss over positive valid steps.
    """
    B,T,M = motor_logits.shape
    if mask is None:
        mask = torch.ones(B,T, device=motor_logits.device)
    pos_mask = ((y_cls > 0).float() * mask)  # [B,T]
    if pos_mask.sum() < 1:
        return motor_logits.new_zeros(())
    # gather logits for pos frames
    idx_b, idx_t = torch.where(pos_mask > 0.5)
    cls = (y_cls[idx_b, idx_t] - 1).long()              # 0..M-1
    logits_sel = motor_logits[idx_b, idx_t, :]          # [N,M]
    # Additive margin on true class
    rows = torch.arange(logits_sel.size(0), device=motor_logits.device)
    logits_m = logits_sel.clone()
    logits_m[rows, cls] -= margin
    logits_m = logits_m * scale
    logp = F.log_softmax(logits_m, dim=-1)
    p    = logp.exp()
    p_y  = p[rows, cls]
    logp_y = logp[rows, cls]
    focal = -(alpha * ((1.0 - p_y).clamp(1e-6,1.0)**gamma) * logp_y)
    return focal.mean()

# =============== Real-time latch inference (validation) ===============
def _majority_int(vals: list):
    if len(vals) == 0: return None
    tensor = torch.tensor(vals)
    uniq, cnt = torch.unique(tensor, return_counts=True)
    j = int(torch.argmax(cnt).item())
    return int(uniq[j].item()), int(cnt[j].item())

def realtime_latch_gate_motor(gate_logits: torch.Tensor, motor_logits: torch.Tensor,
                              theta=0.65, kn_k=3, kn_nw=5, vote_n=5):
    """
    gate_logits:  [B,T]
    motor_logits: [B,T,M]
    Returns predictions [B,T] in {0..M} under absorbing latch policy:
      - Before latch: predict 0 (normal)
      - Latch trigger: if in last N frames there are >=K frames with sigmoid(g)>=theta
      - Latched motor: majority over argmax(motor_logits) in last `vote_n` frames at trigger
      - After latch: always output that motor class (1..M), never return to 0
    """
    B,T = gate_logits.shape
    M = motor_logits.shape[-1]
    preds = torch.zeros(B, T, dtype=torch.long, device=gate_logits.device)
    g = torch.sigmoid(gate_logits)  # [B,T]
    for b in range(B):
        latched = False
        latched_k = 0
        gate_hits = []
        motor_hist = []
        for t in range(T):
            p = g[b,t].item()
            gate_hits.append(1 if p >= theta else 0)
            if len(gate_hits) > kn_nw: gate_hits.pop(0)
            # motor argmax at this frame (candidate always exists)
            k_hat = int(torch.argmax(motor_logits[b,t]).item())
            motor_hist.append(k_hat)
            if len(motor_hist) > vote_n: motor_hist.pop(0)
            if not latched:
                if sum(gate_hits) >= kn_k:
                    # select motor by majority among recent motor_hist
                    m_star, _ = _majority_int(motor_hist)
                    latched, latched_k = True, m_star
                    preds[b,t] = latched_k + 1
                else:
                    preds[b,t] = 0
            else:
                preds[b,t] = latched_k + 1
    return preds  # [B,T] in {0..M}

# ================================ Main ================================
def main():
    # Device
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print("Using device:", device)

    # Input
    try:
        link_count = int(input("How many links?: ").strip())
    except Exception:
        link_count = 1; print("Invalid input. Using link_count=1.")

    # Load data
    data_path = os.path.join("data_storage", f"link_{link_count}", "fault_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    dset   = np.load(data_path, allow_pickle=True)
    d_rel  = dset["desired_link_rel"]   # (S,T,L,4,4)
    a_rel  = dset["actual_link_rel"]
    labels = dset["label"]              # (S,T,M) with 1=normal, 0=fault
    S,T,L  = d_rel.shape[:3]
    M      = labels.shape[2]

    # Features & targets
    X = build_features_twist_only(d_rel, a_rel)           # (S,T,30*L)
    y_bin_np = (1.0 - labels).astype(np.float32)          # 1=fault, 0=normal
    FEAT = X.shape[2]
    print(f"Loaded dataset: S={S}, T={T}, L={L}, M={M}, feature_dim={FEAT} (TwistOnly)")

    # Sanity: at most one fault per frame
    if (y_bin_np.sum(axis=2) > 1.0).any():
        raise ValueError("Found frames with >1 fault per frame; this script assumes max one.")

    # Split + normalize
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    X_all = torch.from_numpy(X).float()
    y_bin = torch.from_numpy(y_bin_np).float()
    n_tr  = int(0.8*S); n_va = S - n_tr
    train_ds_i, val_ds_i = random_split(TensorDataset(X_all, y_bin), [n_tr, n_va],
                                        generator=torch.Generator().manual_seed(SEED))
    tr_idx = train_ds_i.indices; va_idx = val_ds_i.indices
    mu  = X_all[tr_idx].reshape(-1, FEAT).mean(dim=0)
    std = X_all[tr_idx].reshape(-1, FEAT).std(dim=0) + 1e-6
    X_all = (X_all - mu)/std
    train_ds = TensorDataset(X_all[tr_idx], y_bin[tr_idx])
    val_ds   = TensorDataset(X_all[va_idx], y_bin[va_idx])

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SEQ, shuffle=True, drop_last=False,
        collate_fn=lambda b: collate_multi_prefix(b, VIEWS_PER_SEQ, MIN_WIN, LOOKBACK_CAP, POS_BIAS_P)
    )
    val_loader = DataLoader(
        val_ds, batch_size=VAL_BS, shuffle=False, drop_last=False,
        collate_fn=collate_full_sequence
    )

    # Model
    if MODEL_TYPE.lower()=="transformer":
        model = FaultTransformerGateMotor(
            input_dim=FEAT, motors=M,
            d_model=(192 if WIDER_MODEL else 128),
            heads=8, layers=(5 if WIDER_MODEL else 4),
            ff=(384 if WIDER_MODEL else 256),
            drop=0.1, causal=True
        ).to(device)
        save_dir = "Transformer"
    else:
        model = FaultTCNGateMotor(
            input_dim=FEAT, motors=M,
            hidden=(256 if WIDER_MODEL else 128),
            layers=(7 if WIDER_MODEL else 6),
            k=(5 if WIDER_MODEL else 3),
            p=0.1
        ).to(device)
        save_dir = "TCN"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # AMP
    amp_dtype = torch.float16 if device.type=="mps" else (torch.bfloat16 if device.type=="cpu" else torch.float16)
    use_scaler = (device.type=="cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # -------- Train loop --------
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{'Transformer' if MODEL_TYPE=='transformer' else 'TCN'}_link_{link_count}_TwistOnly_GateLatch_FINAL.pth")

    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss_sum=0.0; tr_steps=0.0

        for xb, yb_win, mb in train_loader:
            xb, yb_win, mb = xb.to(device), yb_win.to(device), mb.to(device)  # xb:[B,L,D], yb_win:[B,L,M]
            y_any = (yb_win.sum(dim=2) > 0.5).float()                        # [B,L] 1 if any motor faulty

            optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    gate_logits, motor_logits = model(xb, mb)  # [B,L], [B,L,M]
                    # Loss components
                    L_gate = gate_weighted_bce_with_logits(gate_logits, y_any, w_pos=GATE_POS_W, w_neg=GATE_NEG_W, mask=mb)
                    L_rev  = gate_antireversion_penalty(gate_logits, y_any, mask=mb, lam=LAMBDA_REV)
                    # Motor loss only on positive frames
                    y_cls  = ybin_to_class(yb_win)                                # [B,L] 0..M
                    L_mot  = motor_focal_margin_loss_framewise(motor_logits, y_cls, mask=mb,
                                                               gamma=MOTOR_GAMMA, margin=MOTOR_MARGIN,
                                                               scale=MOTOR_SCALE, alpha=MOTOR_ALPHA)
                    loss = LAMBDA_GATE*L_gate + L_rev + LAMBDA_MOTOR*L_mot
                if use_scaler:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            else:
                gate_logits, motor_logits = model(xb, mb)
                L_gate = gate_weighted_bce_with_logits(gate_logits, y_any, w_pos=GATE_POS_W, w_neg=GATE_NEG_W, mask=mb)
                L_rev  = gate_antireversion_penalty(gate_logits, y_any, mask=mb, lam=LAMBDA_REV)
                y_cls  = ybin_to_class(yb_win)
                L_mot  = motor_focal_margin_loss_framewise(motor_logits, y_cls, mask=mb,
                                                           gamma=MOTOR_GAMMA, margin=MOTOR_MARGIN,
                                                           scale=MOTOR_SCALE, alpha=MOTOR_ALPHA)
                loss = LAMBDA_GATE*L_gate + L_rev + LAMBDA_MOTOR*L_mot
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

            tr_loss_sum += loss.item()*mb.sum().item(); tr_steps += mb.sum().item()

        train_loss = tr_loss_sum / max(tr_steps,1.0)

        # ---- validation (full sequences) ----
        model.eval()
        val_loss_sum=0.0; val_steps=0.0
        preds_all=[]; trues_all=[]; masks_all=[]

        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                y_any = (yb.sum(dim=2) > 0.5).float()
                if USE_AMP:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        gate_logits, motor_logits = model(xb, mb)   # [B,T], [B,T,M]
                        # proxy val loss (same components, full seq)
                        L_gate = gate_weighted_bce_with_logits(gate_logits, y_any, w_pos=GATE_POS_W, w_neg=GATE_NEG_W, mask=mb)
                        L_rev  = gate_antireversion_penalty(gate_logits, y_any, mask=mb, lam=LAMBDA_REV)
                        y_cls  = ybin_to_class(yb)
                        L_mot  = motor_focal_margin_loss_framewise(motor_logits, y_cls, mask=mb,
                                                                   gamma=MOTOR_GAMMA, margin=MOTOR_MARGIN,
                                                                   scale=MOTOR_SCALE, alpha=MOTOR_ALPHA)
                        vloss  = LAMBDA_GATE*L_gate + L_rev + LAMBDA_MOTOR*L_mot
                else:
                    gate_logits, motor_logits = model(xb, mb)
                    L_gate = gate_weighted_bce_with_logits(gate_logits, y_any, w_pos=GATE_POS_W, w_neg=GATE_NEG_W, mask=mb)
                    L_rev  = gate_antireversion_penalty(gate_logits, y_any, mask=mb, lam=LAMBDA_REV)
                    y_cls  = ybin_to_class(yb)
                    L_mot  = motor_focal_margin_loss_framewise(motor_logits, y_cls, mask=mb,
                                                               gamma=MOTOR_GAMMA, margin=MOTOR_MARGIN,
                                                               scale=MOTOR_SCALE, alpha=MOTOR_ALPHA)
                    vloss  = LAMBDA_GATE*L_gate + L_rev + LAMBDA_MOTOR*L_mot

                val_loss_sum += vloss.item()*mb.sum().item()
                val_steps    += mb.sum().item()

                # LATCH inference for metrics
                pred_cls = realtime_latch_gate_motor(gate_logits, motor_logits,
                                                     theta=LATCH_THETA, kn_k=LATCH_KN_K,
                                                     kn_nw=LATCH_KN_NW, vote_n=LATCH_MOTOR_VOTE_N)  # [B,T]
                preds_all.append(pred_cls.cpu())
                trues_all.append(ybin_to_class(yb.cpu()))
                masks_all.append(mb.cpu())

        val_loss = val_loss_sum / max(val_steps,1.0)
        scheduler.step(val_loss)

        preds = torch.cat(preds_all,0)   # [B,T]
        trues = torch.cat(trues_all,0)   # [B,T]
        masks = torch.cat(masks_all,0)   # [B,T]

        correct = ((preds==trues).float()*masks).sum().item()
        total   = masks.sum().item()
        acc_raw = correct/(total+1e-9)

        is_bg = (trues==0).float()*masks
        is_fg = (trues>0).float()*masks
        bg_acc = (((preds==0).float()*is_bg).sum().item()) / (is_bg.sum().item()+1e-9)
        fg_rec = (((preds>0).float()*is_fg).sum().item()) / (is_fg.sum().item()+1e-9)

        print(f"[{ep:03d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"Top1Acc RAW={acc_raw:.4f} | BGacc={bg_acc:.4f} | FGrec={fg_rec:.4f} | "
              f"theta={LATCH_THETA:.2f} K/N={LATCH_KN_K}/{LATCH_KN_NW}")

    # -------- Save once after all epochs --------
    torch.save({
        "model_state_dict": model.state_dict(),
        "train_mean": mu.cpu().numpy(),
        "train_std":  std.cpu().numpy(),
        "input_dim":  FEAT, "num_motors": M,
        "link_count": link_count, "feature_mode": "TwistOnly",
        "model_type": MODEL_TYPE,
        "gate": {"pos_w": GATE_POS_W, "neg_w": GATE_NEG_W, "lambda_gate": LAMBDA_GATE, "lambda_rev": LAMBDA_REV},
        "motor": {"gamma": MOTOR_GAMMA, "margin": MOTOR_MARGIN, "scale": MOTOR_SCALE, "lambda_motor": LAMBDA_MOTOR},
        "latch": {"theta": LATCH_THETA, "K": LATCH_KN_K, "Nw": LATCH_KN_NW, "vote_n": LATCH_MOTOR_VOTE_N}
    }, save_path)
    print(f"Training complete. Saved final model to {save_path}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
HierFDI (Coupled40D causal + dF/dt) Streaming Visualizer — TRAIN-MATCHED

- Loads HierFDI checkpoint saved by the training script.
- Rebuilds the exact same Coupled40D(causal) features (includes dF/dt) with the same wrench observer.
- Uses saved global Z-Norm (mu, std) from the checkpoint (D=40).
- Runs causal, online inference; latches motor fault using K-of-N on per-motor posterior probs.
- Left: 3D LASDRA links with 8 motors/link (spinning props). Faulted motor turns red.
- Right: bar chart per link (GT fault=1 vs Pred prob).  <-- GT is time-localized (pre-fault = BG).

Usage example
-------------
python3 visualize_stream_hierfdi.py \
  --ckpt FDI_MultiStage/link_3/FDI_MultiStage_ALL.pth \
  --npz  data_storage/link_3/fault_dataset.npz \
  --seq_idx 544 \
  --dt 0.01 --data_hz 100 --speed 1.0 \
  --label_fault_is_one 0 \
  --motors_per_link 8 \
  --theta_m 0.75 --kn_k 3 --kn_nw 8 --vote_n 3
"""

import os, sys, math, argparse, random, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

# ============================================================
#                 Seeds / Device utilities
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_ok = getattr(torch.backends, "mps", None)
    if mps_ok is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ============================================================
#        Basic SE(3) math (훈련 코드와 동일)
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
#    External wrench observer (훈련과 동일)
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
# ============================================================

def build_coupled40_for_one_sequence(
    d_rel, a_rel, labels_motor, dt: float,
    masses=None, inertias=None, fu=None, tau=None, omega=None, vel=None, Rm=None,
    K1_val=0.05, K2_val=0.10
):
    """
    입력(한 시퀀스):
      d_rel: (T, L, 4, 4)
      a_rel: (T, L, 4, 4)
      labels_motor: (T, 8L) motor one-hot (BG 없음)  # for meta only
    반환:
      X: (T, L, 40) = [Fm1, Fi, Fp1, Fi-Fm1, Fi-Fp1, dF/dt, u4(=thrust, τxyz)]
      meta: dict (for pose, etc.)
    """
    T, L = d_rel.shape[:2]
    if masses is None:   masses   = np.ones((L,), dtype=np.float64)
    if inertias is None: inertias = np.tile(np.eye(3, dtype=np.float64)[None,...], (L,1,1))
    if Rm is None:       Rm       = a_rel[..., :3, :3]

    # 속도/각속도 유도 (causal FD)
    if vel is None or omega is None:
        p = a_rel[..., :3, 3]
        vel = np.zeros_like(p)
        omega = np.zeros((T, L, 3), dtype=np.float64)
        for i in range(L):
            R_i = Rm[:, i, :, :]
            p_i = p[:, i, :]
            v_fd = np.zeros_like(p_i); v_fd[1:] = (p_i[1:] - p_i[:-1]) / max(dt, 1e-6)
            rvec = _so3_log(R_i)
            w_fd = np.zeros_like(rvec); w_fd[1:] = (rvec[1:] - rvec[:-1]) / max(dt, 1e-6)
            vel[:, i, :] = v_fd
            omega[:, i, :] = w_fd

    if fu is None:  fu  = np.zeros((T, L))
    if tau is None: tau = np.zeros((T, L, 3))

    K1 = np.eye(6)*float(K1_val)
    K2 = np.eye(6)*float(K2_val)

    # per-link wrench 추정
    F_all = np.zeros((T, L, 6), dtype=np.float64)
    for i in range(L):
        m_i = float(masses[i]); I_i = inertias[i]
        v_i = vel[:, i, :]
        w_i = omega[:, i, :]
        R_i = Rm[:, i, :, :]
        fu_i = fu[:, i] if fu.ndim==2 else fu[:, i, :]
        tau_i = tau[:, i, :]
        Fhat = batch_estimate_wrench_per_link(m_i, I_i, v_i, w_i, fu_i, tau_i, R_i, dt, K1, K2)
        F_all[:, i, :] = Fhat

    # dF/dt (forward diff, 첫 프레임 0)
    dF = np.zeros_like(F_all)
    dF[1:] = (F_all[1:] - F_all[:-1]) / max(dt, 1e-6)

    # u4 구성
    if fu.ndim == 2:
        thrust = fu                    # (T,L)
        u4 = np.stack([thrust, tau[:, :, 0], tau[:, :, 1], tau[:, :, 2]], axis=-1)  # (T,L,4)
    else:
        thrust = fu[:, :, 2] if fu.shape[-1]==3 else fu[:, :, 0]
        u4 = np.stack([thrust, tau[:, :, 0], tau[:, :, 1], tau[:, :, 2]], axis=-1)

    X = np.zeros((T, L, 40), dtype=np.float32)
    for t in range(T):
        for i in range(L):
            Fm1 = F_all[t, i-1, :] if i-1>=0 else np.zeros(6)
            Fi  = F_all[t, i,   :]
            Fp1 = F_all[t, i+1, :] if i+1<L  else np.zeros(6)
            dFi = dF[t, i, :]
            feat = np.concatenate([Fm1, Fi, Fp1, Fi-Fm1, Fi-Fp1, dFi, u4[t, i]], axis=0)  # 40
            X[t, i, :] = feat.astype(np.float32)

    meta = {"L": L}
    return X, meta

# ============================================================
#     State / temporal feature helpers (MultiStage cascade)
# ============================================================
def build_state_features_single(d_rel, a_rel, dt):
    """
    Returns per-link state features [T, L, 36] matching FDI_MultiStage training.
    """
    d_rel = d_rel.astype(np.float32, copy=False)
    a_rel = a_rel.astype(np.float32, copy=False)
    T, L = d_rel.shape[:2]
    p_d = d_rel[..., :3, 3]
    p_a = a_rel[..., :3, 3]
    R_d = d_rel[..., :3, :3]
    R_a = a_rel[..., :3, :3]
    r_d = _so3_log(R_d)
    r_a = _so3_log(R_a)

    def fd(x):
        y = np.zeros_like(x, dtype=np.float32)
        y[1:] = (x[1:] - x[:-1]) / max(dt, 1e-6)
        return y.astype(np.float32)

    v_d = fd(p_d); v_a = fd(p_a)
    a_d = fd(v_d); a_a = fd(v_a)
    omg_d = fd(r_d); omg_a = fd(r_a)

    pe = (p_d - p_a).astype(np.float32)
    ve = (v_d - v_a).astype(np.float32)
    ae = (a_d - a_a).astype(np.float32)
    oe = (omg_d - omg_a).astype(np.float32)

    per_link = np.concatenate([
        p_d, p_a, pe,
        v_d, v_a, ve,
        a_d, a_a, ae,
        omg_d, omg_a, oe
    ], axis=-1).astype(np.float32)  # (T,L,36)
    return per_link

def build_l2_seq_feats(per_link_feat, t_center, win):
    """
    per_link_feat: (T,L,36) -> (win, L*4) with left padding (norms of errors).
    """
    T, L, _ = per_link_feat.shape
    if T == 0:
        return np.zeros((win, L*4), dtype=np.float32)
    t1 = max(0, t_center - win//2)
    t2 = min(T, t1 + win)

    pe = per_link_feat[..., 6:9]
    ve = per_link_feat[..., 15:18]
    ae = per_link_feat[..., 24:27]
    oe = per_link_feat[..., 33:36]

    def norms(x):
        return np.linalg.norm(x, axis=-1)

    f_link_t = np.stack([norms(pe), norms(ve), norms(ae), norms(oe)], axis=-1)  # (T,L,4)
    seg = f_link_t[t1:t2].astype(np.float32)
    Wseg = seg.shape[0]
    if Wseg < win:
        pad = np.zeros((win-Wseg, L, 4), dtype=np.float32)
        seg = np.concatenate([pad, seg], axis=0)
    return seg.reshape(win, L*4).astype(np.float32)

def build_wrench_seq_feats_for_link_segment(raw_dict, link_idx, t_center, win, dt):
    """
    raw_dict must contain:
        a_rel (1,T,L,4,4), d_rel (1,T,L,4,4),
        mass (L,), inertia (L,3,3),
        fu (1,T,L or 1,T,L,3), tau (1,T,L,3)
    Returns (win,16) sequence with left padding.
    """
    a_rel = raw_dict["a_rel"][0]
    d_rel = raw_dict["d_rel"][0]
    masses = raw_dict["mass"]
    inertias = raw_dict["inertia"]
    fu = raw_dict["fu"]
    tau = raw_dict["tau"]

    T = a_rel.shape[0]
    t1 = max(0, t_center - win//2)
    t2 = min(T, t1 + win)

    p = a_rel[:, :, :3, 3]
    v_fd = np.zeros_like(p, dtype=np.float32)
    v_fd[1:] = (p[1:] - p[:-1]) / max(dt, 1e-6)
    rvec = _so3_log(a_rel[:, :, :3, :3])
    w_fd = np.zeros_like(rvec, dtype=np.float32)
    w_fd[1:] = (rvec[1:] - rvec[:-1]) / max(dt, 1e-6)

    m_i = float(masses[link_idx])
    I_i = inertias[link_idx].astype(np.float32)
    v_i = v_fd[:, link_idx, :].astype(np.float32)
    w_i = w_fd[:, link_idx, :].astype(np.float32)
    R_i = a_rel[:, link_idx, :3, :3].astype(np.float32)

    if fu.ndim == 3:
        fu_i = fu[0, :, link_idx].astype(np.float32)
    else:
        fu_i = fu[0, :, link_idx, :].astype(np.float32)
    tau_i = tau[0, :, link_idx, :].astype(np.float32)

    K1 = np.eye(6, dtype=np.float32) * 0.05
    K2 = np.eye(6, dtype=np.float32) * 0.10

    Fhat = batch_estimate_wrench_per_link(m_i, I_i, v_i, w_i, fu_i, tau_i, R_i, dt, K1, K2)
    dF = np.zeros_like(Fhat, dtype=np.float32)
    dF[1:] = (Fhat[1:] - Fhat[:-1]) / max(dt, 1e-6)

    if fu.ndim == 3:
        thrust = fu[0, :, link_idx].astype(np.float32)
    else:
        thrust = (fu[0, :, link_idx, 2] if fu.shape[-1] >= 3 else fu[0, :, link_idx, 0]).astype(np.float32)
    u4 = np.stack([
        thrust,
        tau[0, :, link_idx, 0],
        tau[0, :, link_idx, 1],
        tau[0, :, link_idx, 2]
    ], axis=-1).astype(np.float32)

    seg_F = Fhat[t1:t2]
    seg_dF = dF[t1:t2]
    seg_u = u4[t1:t2]
    Wseg = seg_F.shape[0]
    if Wseg < win:
        padF = np.zeros((win-Wseg, 6), dtype=np.float32)
        paddF = np.zeros((win-Wseg, 6), dtype=np.float32)
        padu = np.zeros((win-Wseg, 4), dtype=np.float32)
        seg_F = np.concatenate([padF, seg_F], axis=0)
        seg_dF = np.concatenate([paddF, seg_dF], axis=0)
        seg_u = np.concatenate([padu, seg_u], axis=0)

    seq = np.concatenate([seg_F, seg_dF, seg_u], axis=-1).astype(np.float32)  # (win,16)
    return seq

@torch.no_grad()
def l1_predict_onset_for_sequence(model, ft_seq, device, win, theta, step=1):
    """
    ft_seq: (T, D_total) float32 numpy.
    Returns onset frame index or -1.
    """
    model.eval()
    T = ft_seq.shape[0]
    if T < win:
        return -1
    windows = []
    ends = []
    for t in range(win-1, T, step):
        windows.append(ft_seq[t-win+1:t+1, :])
        ends.append(t)
    xb = torch.from_numpy(np.stack(windows, 0)).float().to(device)
    logits = model(xb).squeeze(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    for q, t in zip(probs, ends):
        if q > theta:
            return int(t)
    return -1

@torch.no_grad()
def l2_predict_logits(model, x_seq, device):
    xb = torch.from_numpy(x_seq[None, ...]).float().to(device)
    logits = model(xb)
    return logits.squeeze(0).cpu()

def _infer_lstm_layers(state_dict, prefix):
    return len([k for k in state_dict.keys() if k.startswith(prefix) and "_reverse" not in k])

def _infer_bidirectional(state_dict, prefix):
    return any(k.startswith(prefix + "0_reverse") for k in state_dict.keys())

class MultiStageCascadeRunner:
    def __init__(self, ckpt, device, meta, motors_per_link, dt):
        self.device = device
        self.meta = meta
        self.raw = meta["raw"]
        self.dt = float(dt)
        self.L = int(meta["L"])
        self.MPL = int(motors_per_link)
        self.motors_total = self.L * self.MPL

        d_rel = self.raw["d_rel"]
        a_rel = self.raw["a_rel"]
        self.T = d_rel.shape[0]

        per_link_feat = build_state_features_single(d_rel, a_rel, self.dt)
        znorm = ckpt.get("znorm", {})
        mu = np.array(znorm.get("mu"), dtype=np.float32) if isinstance(znorm, dict) and "mu" in znorm else None
        std = np.array(znorm.get("std"), dtype=np.float32) if isinstance(znorm, dict) and "std" in znorm else None
        if mu is None or std is None:
            flat = per_link_feat.reshape(-1, per_link_feat.shape[-1])
            mu = flat.mean(axis=0).astype(np.float32)
            std = (flat.std(axis=0) + 1e-6).astype(np.float32)
        self.mu = mu.reshape(1, 1, -1)
        self.std = std.reshape(1, 1, -1)
        self.per_link_norm = ((per_link_feat - self.mu) / self.std).astype(np.float32)
        self.ft_seq = self.per_link_norm.reshape(self.T, -1)

        self.l1_info = ckpt.get("L1", {})
        state_l1 = self.l1_info.get("state_dict")
        if state_l1 is None:
            raise KeyError("MultiStage checkpoint missing L1.state_dict")
        model_type = str(self.l1_info.get("model_type", "bilstm")).lower()
        if model_type != "bilstm":
            raise NotImplementedError(f"L1 model_type '{model_type}' not supported in visualizer yet.")
        self.win1 = int(self.l1_info.get("window", 100))
        self.theta1 = float(self.l1_info.get("threshold", 0.6))
        hidden_l1 = state_l1["lstm.weight_ih_l0"].shape[0] // 4
        in_dim_l1 = state_l1["lstm.weight_ih_l0"].shape[1]
        layers_l1 = _infer_lstm_layers(state_l1, "lstm.weight_ih_l")
        dropout_l1 = float(self.l1_info.get("dropout", 0.30))
        self.l1_model = L1BiLSTM(in_dim_l1, hidden_l1, layers_l1, dropout_l1).to(device)
        self.l1_model.load_state_dict(state_l1, strict=True)
        self.l1_model.eval()

        # L2
        l2_info = ckpt.get("L2", {})
        state_l2 = l2_info.get("state_dict")
        if state_l2 is not None:
            w0 = state_l2["enc.rnn.weight_ih_l0"]
            in_dim_l2 = w0.shape[1]
            hidden_l2 = w0.shape[0] // 4
            layers_l2 = _infer_lstm_layers(state_l2, "enc.rnn.weight_ih_l")
            bidir_l2 = _infer_bidirectional(state_l2, "enc.rnn.weight_ih_l")
            dropout_l2 = float(l2_info.get("dropout", 0.20))
            out_dim_l2 = int(l2_info.get("num_links", self.L))
            self.l2_model = L2Temporal(in_dim=in_dim_l2, hidden=hidden_l2, layers=layers_l2,
                                       dropout=dropout_l2, out_dim=out_dim_l2, bidir=bidir_l2).to(device)
            self.l2_model.load_state_dict(state_l2, strict=True)
            self.l2_model.eval()
            self.win2 = int(l2_info.get("window", 20))
        else:
            self.l2_model = None
            self.win2 = 20

        # L3~L5 models
        self.win35 = int(ckpt.get("L35", {}).get("WINDOW", 20)) if isinstance(ckpt.get("L35"), dict) else 20
        self.bin_models = {}
        for key in ["L3", "L4A", "L4B", "L5_01", "L5_23", "L5_45", "L5_67"]:
            info = ckpt.get(key, {})
            state = info.get("state_dict")
            if state is None:
                continue
            w = state["enc.rnn.weight_ih_l0"]
            in_dim = w.shape[1]
            hidden = w.shape[0] // 4
            layers = _infer_lstm_layers(state, "enc.rnn.weight_ih_l")
            bidir = _infer_bidirectional(state, "enc.rnn.weight_ih_l")
            dropout = float(info.get("dropout", 0.20))
            model = TemporalBinModel(in_dim=in_dim, hidden=hidden, layers=layers,
                                     dropout=dropout, bidir=bidir).to(device)
            model.load_state_dict(state, strict=True)
            model.eval()
            self.bin_models[key] = model
        # Prepared raw dict for L3 sections
        self.raw_dict = {
            "a_rel": self.raw["a_rel"][None, ...],
            "d_rel": self.raw["d_rel"][None, ...],
            "mass": self.raw["mass"],
            "inertia": self.raw["inertia"],
            "fu": self.raw["fu"],
            "tau": self.raw["tau"]
        }            

        self.detected = False
        self.onset_idx = self.T
        self.latest_link_probs = np.ones(self.L, dtype=np.float32) / max(self.L, 1)
        self.latest_motor_probs = np.zeros(self.MPL, dtype=np.float32)
        self.latest_link_sel = 0

    def _window_with_pad(self, seq, end_idx, win):
        start = max(0, end_idx - win + 1)
        window = seq[start:end_idx+1]
        if window.shape[0] < win:
            pad_val = seq[0:1]
            pad = np.repeat(pad_val, win - window.shape[0], axis=0)
            window = np.concatenate([pad, window], axis=0)
        return window.astype(np.float32)

    def _compute_l1_score(self, t_idx):
        if t_idx < self.win1 - 1:
            return 0.0
        window = self._window_with_pad(self.ft_seq, t_idx, self.win1)
        xb = torch.from_numpy(window[None, ...]).float().to(self.device)
        with torch.no_grad():
            score = torch.sigmoid(self.l1_model(xb)).item()
        return float(score)

    def _compute_l2_probs(self, t_idx):
        if self.l2_model is None:
            return np.ones(self.L, dtype=np.float32) / max(self.L, 1)
        x2_seq = build_l2_seq_feats(self.per_link_norm, t_idx, self.win2)
        logits = l2_predict_logits(self.l2_model, x2_seq, self.device)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
        if probs.shape[0] != self.L:
            probs = np.pad(probs, (0, max(0, self.L - probs.shape[0])), constant_values=0.0)[:self.L]
        s = probs.sum()
        if s <= 0:
            probs = np.ones(self.L, dtype=np.float32) / max(self.L, 1)
        else:
            probs /= s
        return probs

    def _compute_motor_probs(self, t_idx, link_sel):
        if "L3" not in self.bin_models:
            return np.zeros(self.MPL, dtype=np.float32)
        x35 = build_wrench_seq_feats_for_link_segment(self.raw_dict, link_sel, t_idx, self.win35, self.dt)
        xb35 = torch.from_numpy(x35[None, ...]).float().to(self.device)
        with torch.no_grad():
            p3 = torch.softmax(self.bin_models["L3"](xb35)[0], dim=-1).cpu().numpy()
            def _bin_prob(key):
                model = self.bin_models.get(key)
                if model is None:
                    return np.array([0.5, 0.5], dtype=np.float32)
                return torch.softmax(model(xb35)[0], dim=-1).cpu().numpy()
            p4A = _bin_prob("L4A")
            p4B = _bin_prob("L4B")
            p5_01 = _bin_prob("L5_01")
            p5_23 = _bin_prob("L5_23")
            p5_45 = _bin_prob("L5_45")
            p5_67 = _bin_prob("L5_67")
        p_motor_link = np.zeros(self.MPL, dtype=np.float32)
        g03 = p3[0]
        g47 = p3[1] if p3.shape[0] > 1 else 0.0
        g01 = g03 * p4A[0]
        g23 = g03 * p4A[1]
        g45 = g47 * p4B[0]
        g67 = g47 * p4B[1]
        p_motor_link[0] = g01 * p5_01[0]
        p_motor_link[1] = g01 * p5_01[1]
        p_motor_link[2] = g23 * p5_23[0]
        p_motor_link[3] = g23 * p5_23[1]
        p_motor_link[4] = g45 * p5_45[0]
        p_motor_link[5] = g45 * p5_45[1]
        p_motor_link[6] = g67 * p5_67[0]
        p_motor_link[7] = g67 * p5_67[1]
        s = p_motor_link.sum()
        if s > 0:
            p_motor_link /= s
        return p_motor_link
    def step(self, t_idx):
        t_idx = int(min(max(t_idx, 0), self.T - 1))
        score = self._compute_l1_score(t_idx)
        if score >= self.theta1 and not self.detected:
            self.detected = True
            self.onset_idx = t_idx
        if self.detected:
            self.latest_link_probs = self._compute_l2_probs(t_idx)
            self.latest_link_sel = int(np.argmax(self.latest_link_probs))
            self.latest_motor_probs = self._compute_motor_probs(t_idx, self.latest_link_sel)
        else:
            self.latest_link_probs = np.ones(self.L, dtype=np.float32) / max(self.L, 1)
            self.latest_motor_probs = np.zeros(self.MPL, dtype=np.float32)
            self.latest_link_sel = int(np.argmax(self.latest_link_probs))
        global_probs = np.zeros(self.motors_total, dtype=np.float32)
        if self.latest_motor_probs.sum() > 0:
            start = self.latest_link_sel * self.MPL
            end = start + self.MPL
            global_probs[start:end] = self.latest_motor_probs
        p_bg = max(0.0, 1.0 - global_probs.sum())
        return p_bg, global_probs
def run_multistage_inference(ckpt, device, meta, motors_per_link, dt):
    runner = MultiStageCascadeRunner(ckpt, device, meta, motors_per_link=motors_per_link, dt=dt)
    p_bg_seq = np.ones(runner.T, dtype=np.float32)
    p_mot_seq = np.zeros((runner.T, runner.motors_total), dtype=np.float32)
    for t in range(runner.T):
        p_bg, p_global = runner.step(t)
        p_bg_seq[t] = p_bg
        p_mot_seq[t] = p_global
    onset_idx = int(runner.onset_idx)

    return {
        "p_bg_seq": p_bg_seq,
        "p_mot_seq": p_mot_seq,
        "onset_idx": onset_idx
    }

# ============================================================
#                     Data loading helpers
# ============================================================

def compose_cum_from_rel(rel_T):  # rel_T: [T, L, 4, 4]
    T, L = rel_T.shape[:2]
    cum = np.zeros((T, L+1, 4, 4), dtype=rel_T.dtype)
    I = np.eye(4, dtype=rel_T.dtype)
    for t in range(T):
        cur = I.copy()
        cum[t, 0] = I
        for l in range(L):
            cur = cur @ rel_T[t, l]
            cum[t, l+1] = cur
    return cum

def _read_possible_onset_from_npz_dict(d, local_idx: int):
    keys_try = [
        "fault_onset_idx", "onset_idx", "fault_start_idx", "fault_t_idx",
        "fault_onset_frame", "onset_frame"
    ]
    for k in keys_try:
        if k in d:
            arr = d[k]
            try:
                val = int(arr[local_idx]) if hasattr(arr, "__len__") else int(arr)
                if val >= 0:
                    return val
            except Exception:
                pass
    return None

def _eligible_paths(path_str: str):
    if os.path.isdir(path_str): return sorted(glob.glob(os.path.join(path_str, "*.npz")))
    if any(ch in path_str for ch in ("*", "?", "[")): return sorted(glob.glob(path_str))
    if os.path.exists(path_str): return [path_str]
    if path_str.endswith(".npz"):
        prefix = path_str[:-4]
        matches = sorted(glob.glob(prefix + "_*.npz"))
        if matches: return matches
    return []

def load_one_sequence_and_build_coupled40(npz_path: str, seq_idx: int, dt: float,
                                          label_fault_is_one: bool,
                                          motors_per_link: int,
                                          K1_val=0.05, K2_val=0.10):
    """
    반환:
      X: (T,L,40), labels_motor(T, 8L), Dcum(T,L+1,4,4), Acum(T,L+1,4,4), meta
    """
    paths = _eligible_paths(npz_path)
    if not paths:
        raise FileNotFoundError(f"No NPZ files found for path '{npz_path}'")

    total = 0
    sizes = []
    for p in paths:
        with np.load(p, allow_pickle=True) as d:
            if "desired_link_rel" not in d: continue
            s = d["desired_link_rel"].shape[0]
            sizes.append((p, s))
            total += s
    if total == 0:
        raise ValueError(f"No sequences found under '{npz_path}'")
    if not (0 <= seq_idx < total):
        raise AssertionError(f"seq_idx {seq_idx} out of range 0..{total-1}")

    # 대상 파일/로컬 인덱스 계산
    offset = 0
    target_path, local_idx = None, None
    for p, s in sizes:
        if seq_idx < offset + s:
            target_path, local_idx = p, seq_idx - offset
            break
        offset += s
    assert target_path is not None

    with np.load(target_path, allow_pickle=True) as d:
        keys = set(d.files)
        need_rel = {"desired_link_rel", "actual_link_rel", "label"}
        if not need_rel.issubset(keys):
            raise KeyError(f"NPZ needs keys {sorted(need_rel)}; got {sorted(keys)}")

        d_rel_all = d["desired_link_rel"]
        a_rel_all = d["actual_link_rel"]
        labels_all= d["label"]

        d_rel = np.array(d_rel_all[local_idx], dtype=np.float32, copy=True)      # (T,L,4,4)
        a_rel = np.array(a_rel_all[local_idx], dtype=np.float32, copy=True)
        labels = np.array(labels_all[local_idx], dtype=np.float32, copy=True)    # (T, 8L) or (T, 1+8L) 등

        if not label_fault_is_one:
            labels = 1.0 - labels

        # motor-only로 통일
        MPL = motors_per_link
        if labels.shape[1] % MPL == 1:
            # BG 포함 → BG 제거
            labels_motor = labels[:, 1:].astype(np.float32)
        elif labels.shape[1] % MPL == 0:
            labels_motor = labels.astype(np.float32)
        else:
            m_guess = (labels.shape[1] // MPL) * MPL
            labels_motor = labels[:, -m_guess:].astype(np.float32)

        L = d_rel.shape[1]

        masses  = np.array(d["mass"], dtype=np.float32, copy=True) if "mass" in d else np.ones((L,), dtype=np.float32)
        inertia = np.array(d["inertia"], dtype=np.float32, copy=True) if "inertia" in d else np.tile(np.eye(3, dtype=np.float32)[None,...], (L,1,1))
        fu_arr   = d["cmd_force"]  if "cmd_force"  in d else None
        tau_arr  = d["cmd_torque"] if "cmd_torque" in d else None
        omega   = np.array(d["omega"], dtype=np.float32, copy=True)      if "omega"      in d else None
        vel     = np.array(d["vel"], dtype=np.float32, copy=True)        if "vel"        in d else None
        Rm      = np.array(d["R_body"], dtype=np.float32, copy=True)     if "R_body"     in d else None

        if fu_arr is not None:
            fu = np.array(fu_arr, dtype=np.float32, copy=True)[local_idx]
        else:
            fu = np.zeros((d_rel.shape[0], L), dtype=np.float32)
        if tau_arr is not None:
            tau = np.array(tau_arr, dtype=np.float32, copy=True)[local_idx]
        else:
            tau = np.zeros((d_rel.shape[0], L, 3), dtype=np.float32)
        if fu.ndim == 2:
            fu_for_builder = fu[None, ...]
        else:
            fu_for_builder = fu[None, ...]
        tau_for_builder = tau[None, ...]

        # (T,L,40)
        X, meta_f = build_coupled40_for_one_sequence(
            d_rel, a_rel, labels_motor, dt,
            masses=masses, inertias=inertia, fu=fu, tau=tau, omega=omega, vel=vel, Rm=Rm,
            K1_val=K1_val, K2_val=K2_val
        )

        # 누적 포즈 (Base 포함 L+1)
        Dcum = compose_cum_from_rel(d_rel)  # (T,L+1,4,4)
        Acum = compose_cum_from_rel(a_rel)

        onset_from_npz = _read_possible_onset_from_npz_dict(d, local_idx)

    meta = {
        "L": meta_f["L"],
        "Dcum": Dcum,
        "Acum": Acum,
        "onset_from_npz": onset_from_npz,
        "raw": {
            "d_rel": d_rel,
            "a_rel": a_rel,
            "mass": masses,
            "inertia": inertia.astype(np.float32, copy=False),
            "fu": fu_for_builder.astype(np.float32, copy=False),
            "tau": tau_for_builder.astype(np.float32, copy=False),
            "omega": omega.astype(np.float32, copy=False) if omega is not None else None,
            "vel": vel.astype(np.float32, copy=False) if vel is not None else None,
            "labels_motor": labels_motor.astype(np.float32, copy=False),
            "dt": dt
        }
    }
    return X.astype(np.float32), labels_motor.astype(np.float32), meta

# ============================================================
#                     HierFDI Model (훈련과 동일)
# ============================================================

class TemporalAttentionCausal(nn.Module):
    """O(T) 누적합 기반 causal attention (훈련 코드와 동일)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h_blth):
        # h_blth: (B, L, T, H)
        B, L, T, H = h_blth.shape
        s = self.attn(h_blth).squeeze(-1)              # (B,L,T)

        # 수치 안정성: per-(B,L) 축에서 최대값 빼고 exp
        s_max = s.max(dim=2, keepdim=True).values      # (B,L,1)
        exp_s = torch.exp(s - s_max)                   # (B,L,T)

        # 누적합으로 causal-softmax 정규화
        cum_exp = torch.cumsum(exp_s, dim=2)           # (B,L,T)
        weighted_h = exp_s.unsqueeze(-1) * h_blth      # (B,L,T,H)
        cum_weighted_h = torch.cumsum(weighted_h, dim=2)  # (B,L,T,H)

        ctx = cum_weighted_h / (cum_exp.unsqueeze(-1) + 1e-12)   # (B,L,T,H)
        h_out = h_blth + ctx
        return h_out

class L1BiLSTM(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers>1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*2, 1))
    def forward(self, x_bw_d):
        h,_ = self.lstm(x_bw_d)
        out = self.head(h[:,-1,:])
        return out.squeeze(-1)

class L11DCNN(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        c = 64
        self.proj = nn.Linear(in_dim, c)
        self.conv1 = nn.Conv1d(c, 128, 5, padding=2)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 1))
    def forward(self, x_bw_d):
        z = F.relu(self.proj(x_bw_d))
        z = z.transpose(1, 2)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = z.mean(dim=2)
        out = self.head(z)
        return out.squeeze(-1)

class L1Transformer(nn.Module):
    def __init__(self, in_dim, hidden, nhead, nlayers, dropout):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead,
                                           dim_feedforward=hidden*4,
                                           dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
    def forward(self, x_bw_d):
        z = self.in_proj(x_bw_d)
        z = self.enc(z)
        z = z[:,-1,:]
        out = self.head(z)
        return out.squeeze(-1)

class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, bidir=True):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidir, dropout=dropout if layers>1 else 0.0)
        self.out_dim = hidden * (2 if bidir else 1)
    def forward(self, x):
        z = self.norm(x)
        h,_ = self.rnn(z)
        return h[:,-1,:]

class TemporalMultiClassHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim))
    def forward(self, z):
        return self.head(z)

class TemporalBinModel(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, bidir=True):
        super().__init__()
        self.enc = TemporalEncoder(in_dim=in_dim, hidden=hidden, layers=layers,
                                   dropout=dropout, bidir=bidir)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.enc.out_dim, 2))
    def forward(self, x):
        z = self.enc(x)
        return self.head(z)

class L2Temporal(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, out_dim, bidir=True):
        super().__init__()
        self.enc = TemporalEncoder(in_dim=in_dim, hidden=hidden,
                                   layers=layers, dropout=dropout, bidir=bidir)
        self.head = TemporalMultiClassHead(self.enc.out_dim, out_dim, dropout)
    def forward(self, x):
        z = self.enc(x)
        return self.head(z)

class HierFDI(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, link_count, dropout=0.3):
        super().__init__()
        self.L = link_count
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.drop = nn.Dropout(dropout)

        # Causal temporal attention (훈련과 동일)
        self.tattn = TemporalAttentionCausal(hidden_dim)

        # Level 1: no-fault(1) + per-link(L)
        self.nofault_head = nn.Sequential(nn.Linear(hidden_dim*self.L, 256), nn.ReLU(), nn.Linear(256, 1))
        self.link_head    = nn.Linear(hidden_dim, 1)   # per-link score -> concat -> softmax

        # Level 2: per-link region(4)
        self.region_head  = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 4)
        )

        # Level 3: per-link, per-region 2-way
        self.motor_head   = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 8))  # reshape -> (4,2)

    def forward(self, x_btld):
        # x: (B,T,L,D)
        B, T, L, D = x_btld.shape
        assert L == self.L, f"L mismatch: model={self.L}, input={L}"
        x_bltd = x_btld.permute(0,2,1,3).contiguous()       # (B,L,T,D)
        x_flat = x_bltd.view(B*L, T, D)                     # (B*L,T,D)

        h, _ = self.lstm(x_flat)                            # (B*L,T,H)
        h = self.drop(h)
        h_blth = h.view(B, L, T, self.hidden_dim)           # (B,L,T,H)

        # Causal attention
        h_blth = self.tattn(h_blth)                         # (B,L,T,H)

        h_bt_l_h = h_blth.permute(0,2,1,3).contiguous()     # (B,T,L,H)

        # Level 1
        link_score = self.link_head(h_bt_l_h).squeeze(-1)   # (B,T,L)
        ctx = h_bt_l_h.reshape(B, T, L*self.hidden_dim)     # (B,T,LH)
        nof_score = self.nofault_head(ctx).squeeze(-1)      # (B,T)
        l1_logits = torch.cat([nof_score.unsqueeze(-1), link_score], dim=-1)  # (B,T,1+L)

        # Level 2
        reg_logits = self.region_head(h_bt_l_h)             # (B,T,L,4)

        # Level 3
        mot_logits_8 = self.motor_head(h_bt_l_h)            # (B,T,L,8)
        mot_logits = mot_logits_8.view(B, T, L, 4, 2)       # (B,T,L,4,2)

        return l1_logits, reg_logits, mot_logits

# ============================================================
#          Motor posterior from hierarchical heads
# ============================================================

def posterior_motor_probs(l1_logits, reg_logits, mot_logits):
    """
    입력:
      l1_logits: (1,T,1+L)
      reg_logits: (1,T,L,4)
      mot_logits: (1,T,L,4,2)
    출력:
      p_bg: (T,)      # class 0 (no-fault)
      p_motor: (T, L*8)   # marginal posterior per motor
    """
    with torch.no_grad():
        B, T, C = l1_logits.shape
        L = C - 1
        p_l1 = F.softmax(l1_logits, dim=-1)          # (1,T,1+L)
        p_bg = p_l1[0, :, 0]                         # (T,)
        p_link = p_l1[0, :, 1:]                      # (T,L)

        p_reg  = F.softmax(reg_logits, dim=-1)[0]    # (T,L,4)
        p_mot  = F.softmax(mot_logits, dim=-1)[0]    # (T,L,4,2)

        # motor index: (link l, region r, pair b) -> j = l*8 + r*2 + b
        motors = torch.zeros(T, L*8, device=l1_logits.device)
        for l in range(L):
            for r in range(4):
                for b in range(2):
                    j = l*8 + r*2 + b
                    motors[:, j] = p_link[:, l] * p_reg[:, l, r] * p_mot[:, l, r, b]
        return p_bg, motors

# ============================================================
#          K-of-N latch (prob 기반, BG 제외)
# ============================================================

@torch.no_grad()
def latch_motor_kofn_probs(
    p_bg: torch.Tensor,          # (T,)
    p_motor: torch.Tensor,       # (T, M)  (M=8L)
    theta_m: float = 0.75,
    kn_k: int = 3,
    kn_nw: int = 8,
    vote_n: int = 3,
):
    """
    반환:
      preds: (T,) where 0=BG, k>0=1..M (모터 인덱스+1; BG=0)
    """
    T, M = p_motor.shape
    top1_prob, top1_idx = p_motor.max(dim=-1)  # (T,), (T,)

    is_fg = (top1_prob >= theta_m) & (p_bg < 0.5)  # BG가 높으면 무시
    hits = is_fg.int()
    csum = torch.cumsum(hits, dim=0)
    csum_left = F.pad(csum, (kn_nw,0))[:-kn_nw]
    win_sum = csum - csum_left
    cond = (win_sum >= kn_k)

    # 최초 만족 시점
    t_idx = torch.arange(T, device=p_motor.device)
    masked = torch.where(cond, t_idx, torch.full_like(t_idx, T))
    t_star = masked.min()
    has = bool(t_star != T)

    preds = torch.zeros(T, dtype=torch.long, device=p_motor.device)
    if not has:
        return preds  # 전구간 BG

    # vote_n 창에서 가장 빈번히 나온 모터 선택
    t0 = max(0, int(t_star.item()) - vote_n + 1)
    sl = slice(t0, int(t_star.item())+1)
    onehot = F.one_hot(top1_idx[sl], num_classes=M).sum(dim=0)
    latched = int(torch.argmax(onehot).item()) + 1  # 1..M (0=BG 예약)
    preds[int(t_star.item()):] = latched
    return preds

# ============================================================
#            GT time-localization utilities
# ============================================================

PREFAULT_WIN      = 60
MIN_BASE_FRAMES   = 10
EPS_STD           = 1e-6

def estimate_onset_from_pose(Dcum, Acum, fps_like_hz=100.0, min_sustain=5, sigma_k=3.0):
    T = Dcum.shape[0]
    pos_d = Dcum[..., :3, 3]
    pos_a = Acum[..., :3, 3]
    err = np.linalg.norm((pos_a - pos_d).reshape(T, -1), axis=1)  # [T]
    win = min(PREFAULT_WIN, T//4 if T>=8 else T)
    mu = err[:win].mean() if win>0 else err.mean()
    std= err[:win].std()  if win>0 else err.std()
    thr = mu + sigma_k * (std + 1e-9)
    above = (err >= thr).astype(np.int32)
    if above.sum()==0:
        return T
    c = 0
    for t in range(T):
        if above[t]:
            c += 1
            if c >= max(1, min_sustain):
                return max(0, t - min_sustain + 1)
        else:
            c = 0
    return T

def localize_gt_labels_motor_only(Y_motor, onset_idx):
    """
    입력: Y_motor [T, M] (모터별 one-hot/확률, BG 없음)
    출력: Y_with_bg [T, 1+M] — onset 이전 BG=1, 이후엔 motor 원-핫
    """
    T, M = Y_motor.shape
    Yloc = np.zeros((T, 1+M), dtype=np.float32)
    if onset_idx >= T:
        Yloc[:, 0] = 1.0
        return Yloc
    Yloc[:onset_idx, 0] = 1.0
    Yloc[onset_idx:, 1:] = (Y_motor[onset_idx:] > 0.5).astype(np.float32)
    return Yloc

# ============================================================
#                           Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz",  required=True)
    ap.add_argument("--seq_idx", type=int, default=0)

    # dataset / features
    ap.add_argument("--dt", type=float, default=0.01, help="feature dt used in observer/FD (train used 0.01)")
    ap.add_argument("--label_fault_is_one", type=int, default=0)

    # label time-localization controls
    ap.add_argument("--labels_are_seqlevel", type=int, default=1,
                    help="1: label이 시퀀스 레벨(항상 같은 모터=1)일 수 있으니 onset을 찾아 GT를 시간정렬")
    ap.add_argument("--gt_onset", type=int, default=-1,
                    help=">=0 이면 GT onset 프레임을 강제로 사용")
    ap.add_argument("--gt_localize_disable", type=int, default=0,
                    help="1이면 기존 동작(시간정렬 안함)")

    # latch (motor posterior)
    ap.add_argument("--theta_m", type=float, default=0.75)
    ap.add_argument("--kn_k", type=int, default=3)
    ap.add_argument("--kn_nw", type=int, default=8)
    ap.add_argument("--vote_n", type=int, default=3)

    # playback / world
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--data_hz", type=float, default=100.0)
    ap.add_argument("--prepend_base", type=int, default=1)
    ap.add_argument("--fix_origin",  type=int, default=1)

    # motors/layout
    ap.add_argument("--motors_per_link", type=int, default=8)
    ap.add_argument("--anchor_ratio", type=float, default=0.85)
    ap.add_argument("--arm_len", type=float, default=0.22)

    # props
    ap.add_argument("--prop_blades", type=int, default=4)
    ap.add_argument("--prop_radius", type=float, default=0.10)
    ap.add_argument("--prop_chord",  type=float, default=0.035)
    ap.add_argument("--prop_alpha",  type=float, default=0.85)
    ap.add_argument("--stem_alpha",  type=float, default=0.95)
    ap.add_argument("--prop_rps", type=float, default=15.0)
    ap.add_argument("--spin_dir_alt", type=int, default=1)

    # video
    ap.add_argument("--save_video", type=int, default=0)
    ap.add_argument("--out", type=str, default="output.mp4")
    ap.add_argument("--video_fps", type=int, default=30)
    ap.add_argument("--codec", type=str, default="libx264")
    ap.add_argument("--bitrate", type=str, default="4000k")
    ap.add_argument("--dpi", type=int, default=150)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    torch.set_float32_matmul_precision("high")
    print("📥 device:", device)

    # ---- load checkpoint (dict expected)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # ---- (A') Robust checkpoint/meta resolver (multi-stage & legacy friendly)
    def _get_path(d, path):
        cur = d
        for k in path.split('.'):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
        return cur

    def _find_nested(d, names):
        # names: list of candidate single keys (not dotted)
        stack = [d]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in names:
                        return v
                    if isinstance(v, dict):
                        stack.append(v)
        return None

    def _find_znorm(ckpt):
        # 1) explicit znorm dict with {mean,std}
        z = _get_path(ckpt, "znorm")
        if isinstance(z, dict) and ("mean" in z) and ("std" in z):
            return z["mean"], z["std"]
        # 2) common name pairs
        pairs = [("train_mean","train_std"), ("mu","sigma"), ("norm_mean","norm_std")]
        for a, b in pairs:
            va = _find_nested(ckpt, [a])
            vb = _find_nested(ckpt, [b])
            if va is not None and vb is not None:
                return va, vb
        # 3) deep search any dict that looks like {mean,std}
        stack = [ckpt]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                if "mean" in x and "std" in x:
                    return x["mean"], x["std"]
                for v in x.values():
                    if isinstance(v, dict):
                        stack.append(v)
        return None, None

    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                return ckpt["model_state_dict"], "model_state_dict"
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                return ckpt["state_dict"], "state_dict"
            if "L1" in ckpt and isinstance(ckpt["L1"], dict) and isinstance(ckpt["L1"].get("state_dict"), dict):
                print("⚙️ MultiStage checkpoint detected — using ckpt['L1']['state_dict'].")
                return ckpt["L1"]["state_dict"], "L1.state_dict"
            # raw state_dict?
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return ckpt, "(raw)"
        raise KeyError("Cannot locate a valid state_dict in the checkpoint.")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state, state_where = _extract_state_dict(ckpt)

    # Try to read optional meta (fallbacks are handled later)
    train_mean_opt, train_std_opt = _find_znorm(ckpt)
    D_in_opt       = _find_nested(ckpt, ["input_dim","in_dim","feature_dim","input_size"])
    hidden_opt     = _find_nested(ckpt, ["hidden_dim","H","lstm_hidden","lstm_hidden_dim"])
    layers_opt     = _find_nested(ckpt, ["num_layers","layers","lstm_layers"])
    link_count_opt = _find_nested(ckpt, ["link_count","L","links"])
    dropout_opt    = _find_nested(ckpt, ["dropout","pdrop"])
    feat_mode      = _find_nested(ckpt, ["feature_mode"]) or "coupled40d_dfdt"
    K1_val         = float(_find_nested(ckpt, ["observer_K1"]) or 0.05)
    K2_val         = float(_find_nested(ckpt, ["observer_K2"]) or 0.10)

    print(f"CKPT meta(partial): state={state_where}, "
          f"D_in?={D_in_opt is not None}, znorm?={train_mean_opt is not None}")

    # ---- (B') Load data & build TRAIN-MATCHED features (we can build without meta)
    X, Y_motor_raw, meta = load_one_sequence_and_build_coupled40(
        args.npz, args.seq_idx, dt=args.dt,
        label_fault_is_one=bool(args.label_fault_is_one),
        motors_per_link=args.motors_per_link,
        K1_val=K1_val, K2_val=K2_val
    )
    T, L_data, D_built = X.shape

    is_multistage = (state_where == "L1.state_dict")
    stream_multi = None
    p_mot_seq_np = None
    pred_seq_np = None
    p_bg_seq_np = None
    ckpt_meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    if is_multistage:
        link_count = int(ckpt_meta.get("link_count", meta["L"]))
        motors_per_link = int(ckpt_meta.get("motors_per_link", args.motors_per_link))
        stream_multi = run_multistage_inference(ckpt, device, meta, motors_per_link=motors_per_link, dt=args.dt)
        p_bg_seq_np = stream_multi["p_bg_seq"].astype(np.float32)
        p_mot_seq_np = stream_multi["p_mot_seq"].astype(np.float32)
        onset_auto = stream_multi["onset_idx"]
        link_count_ckpt = link_count
        motors_physical = link_count * motors_per_link
        p_bg_tensor = torch.from_numpy(p_bg_seq_np).float()
        p_mot_tensor = torch.from_numpy(p_mot_seq_np).float()
        pred_seq_np = latch_motor_kofn_probs(
            p_bg_tensor,
            p_mot_tensor,
            theta_m=args.theta_m,
            kn_k=args.kn_k,
            kn_nw=args.kn_nw,
            vote_n=args.vote_n
        )[0].cpu().numpy().astype(np.int64)
    else:
        # ---- (B'2) Infer architecture from state_dict when not provided
        def _infer_arch_from_state(state):
            lstm_wih_keys = [k for k in state.keys() if k.startswith("lstm.weight_ih_l")]
            num_layers = len(lstm_wih_keys) if lstm_wih_keys else 1
            if "lstm.weight_ih_l0" in state:
                wih0 = state["lstm.weight_ih_l0"]
                H = wih0.shape[0] // 4
                D_from_state = wih0.shape[1]
            else:
                H, D_from_state = (hidden_opt or 256), (D_in_opt or D_built)
            # infer L from nofault_head first linear (input = L*H)
            L_infer = None
            nf0w = state.get("nofault_head.0.weight")
            if nf0w is not None and nf0w.ndim == 2:
                in_dim = nf0w.shape[1]
                if (H is not None) and H > 0:
                    L_infer = int(in_dim // H)
            return H, num_layers, D_from_state, L_infer

        H_inf, LAY_inf, D_from_state, L_infer = _infer_arch_from_state(state)

        # Final resolved hyperparams
        D_in          = int(D_in_opt or D_from_state or D_built)
        hidden_dim_ckpt = int(hidden_opt or H_inf or 256)
        num_layers_ckpt = int(layers_opt or LAY_inf or 1)
        link_count_ckpt = int(link_count_opt or L_infer or L_data)
        dropout_ckpt    = float(dropout_opt or 0.30)

        if D_in != D_built:
            raise ValueError(f"[ERROR] Feature dim mismatch: built D={D_built}, ckpt expects D_in={D_in}. "
                             f"(체크포인트의 학습 피처 모드와 현재 시각화 피처가 다릅니다)")

        # ---- (B'3) Resolve / fallback Z-Norm
        if train_mean_opt is None or train_std_opt is None:
            mu = X.reshape(-1, D_built).mean(axis=0)
            sd = X.reshape(-1, D_built).std(axis=0) + 1e-6
            train_mean = mu.astype(np.float32)
            train_std  = sd.astype(np.float32)
            print("🔁 No Z-Norm in ckpt — computed mean/std from the loaded sequence.")
        else:
            train_mean = np.array(train_mean_opt, dtype=np.float32).reshape(-1)
            train_std  = np.array(train_std_opt,  dtype=np.float32).reshape(-1)

        print(f"Resolved: L={link_count_ckpt}, D_in={D_in}, H={hidden_dim_ckpt}, layers={num_layers_ckpt}, "
              f"dropout={dropout_ckpt}, feat_mode={feat_mode}")

        link_count = link_count_ckpt
        motors_per_link = int(args.motors_per_link)
        motors_physical = link_count * motors_per_link

        if X.shape[1] != link_count:
            raise ValueError(f"[ERROR] Link-count mismatch: ckpt L={link_count}, data L={X.shape[1]}. "
                             f"Use a dataset with the same L as training, or retrain.")

        # ---- (E) Instantiate model STRICTLY with ckpt L and load weights
        model = HierFDI(D_in, hidden_dim_ckpt, num_layers_ckpt, link_count=link_count, dropout=dropout_ckpt).to(device)
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            print(f"⚠️ load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:   print("   missing:", missing[:12], "..." if len(missing)>12 else "")
            if unexpected:print("   unexpected:", unexpected[:12], "..." if len(unexpected)>12 else "")
        model.eval()

        mu    = np.array(train_mean, dtype=np.float32).reshape(-1)
        std   = np.array(train_std,  dtype=np.float32).reshape(-1)
        mu_t  = torch.as_tensor(mu, dtype=torch.float32, device=device)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
        std_t = torch.clamp(std_t, min=EPS_STD)
        X_t = torch.from_numpy(X).to(device).float()

        @torch.inference_mode()
        def forward_upto(t_end):
            win = 100  
            start = max(0, t_end - win + 1)  
            x = X_t[start : t_end + 1]   
            x = (x - mu_t) / std_t
            x = x.unsqueeze(0)
            l1, reg, mot = model(x)
            return l1, reg, mot

    # ---- (B'2) Infer architecture from state_dict when not provided
    L = link_count

    # Pose for viz (L+1 with base)
    Dcum, Acum = meta["Dcum"], meta["Acum"]  # (T, L+1, 4,4)
    if args.prepend_base and Acum.shape[1] == (link_count + 2):
        Dcum = Dcum[:, 1:, :, :]
        Acum = Acum[:, 1:, :, :]

    # ---- (D) Build time-localized GT
    if args.gt_localize_disable:
        any_fault = (Y_motor_raw > 0.5).any(axis=1, keepdims=True).astype(np.float32)
        Y_with_bg = np.concatenate([1.0 - any_fault, Y_motor_raw], axis=1)
        onset = None
        print("[GT] localization disabled; BG added from label nonzero.")
    else:
        if is_multistage:
            onset = int(stream_multi["onset_idx"])
            why = "L1-onset(cascade)"
        else:
            onset_cli = int(args.gt_onset) if args.gt_onset is not None else -1
            onset_from_npz = meta.get("onset_from_npz", None)
            if onset_cli >= 0:
                onset = onset_cli; why="CLI(--gt_onset)"
            elif onset_from_npz is not None:
                onset = int(onset_from_npz); why="NPZ(fault_onset_idx)"
            else:
                looks_seqlevel = (Y_motor_raw.std(axis=0).sum() == 0.0) or bool(args.labels_are_seqlevel)
                if looks_seqlevel:
                    onset = estimate_onset_from_pose(Dcum, Acum, fps_like_hz=args.data_hz, min_sustain=5, sigma_k=3.0)
                    why="pose-heuristic(3σ,5frames)"
                else:
                    idx = np.where((Y_motor_raw > 0.5).any(axis=1))[0]
                    onset = int(idx[0]) if idx.size>0 else T
                    why="label(time-varying)"
        Y_with_bg = localize_gt_labels_motor_only(Y_motor_raw, onset)
        print(f"[GT] time-localized with onset={onset} source={why}")

    # =======================================================
    #               Matplotlib figure & axes
    # =======================================================
    rows = link_count
    fig_h = 6 + 1.8 * max(0, rows - 1)
    plt.close("all")
    fig = plt.figure(figsize=(12, fig_h))
    gs = fig.add_gridspec(rows, 2, width_ratios=[3, 2], height_ratios=[1]*rows, wspace=0.35, hspace=0.45)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("LASDRA (desired vs actual)", fontsize=18)

    axbars = [fig.add_subplot(gs[r, 1]) for r in range(rows)]

    # world limits
    sample = Acum[:min(200, T)]
    def positions_from_cum(cum): return cum[..., :3, 3]
    def normalize_by_base(P):
        return P - P[:, :1, :] if P.ndim == 3 else P - P[:1, :]
    p_all = positions_from_cum(sample)
    if args.fix_origin:
        p_all = normalize_by_base(p_all)
    p = p_all.reshape(-1, 3)
    pmin, pmax = p.min(axis=0), p.max(axis=0)
    span = (pmax - pmin).max()
    center = (pmax + pmin)/2
    lim = span*0.8 if span > 0 else 0.5
    ax3d.set_xlim(center[0]-lim, center[0]+lim)
    ax3d.set_ylim(center[1]-lim, center[1]+lim)
    ax3d.set_zlim(center[2]-lim, center[2]+lim)

    desired_lines, actual_lines = [], []
    for _ in range(link_count):
        d_ln, = ax3d.plot([], [], [], linestyle="--", color="g", lw=2.0, alpha=1.0)
        a_ln, = ax3d.plot([], [], [], linestyle="-",  color="k", lw=2.5, alpha=0.45)
        desired_lines.append(d_ln); actual_lines.append(a_ln)

    desired_nodes = ax3d.scatter([], [], [], s=15, c="g", alpha=1.0)
    actual_nodes  = ax3d.scatter([], [], [], s=18, c="k", alpha=0.45)

    base_marker = ax3d.scatter([0], [0], [0], s=120, marker='o',
                               facecolor='k', edgecolor='y', linewidth=2.0, alpha=1.0, zorder=5)
    base_text   = ax3d.text(0.05, 0.05, 0.05, "BASE", color="y", fontsize=10, ha="left", va="bottom")

    # motor visuals (8 per link assumed)
    def _norm(v, eps=1e-9): n = np.linalg.norm(v); return v/(n+eps)

    stems_lines    = [[None]*motors_per_link for _ in range(link_count)]
    blade_patches  = [[[]  for _ in range(motors_per_link)] for _ in range(link_count)]
    fault_texts    = [[None]*motors_per_link for _ in range(link_count)]

    for li in range(link_count):
        for mj in range(motors_per_link):
            ln_stem,  = ax3d.plot([], [], [], color="k", lw=1.2, alpha=args.stem_alpha)
            stems_lines[li][mj] = ln_stem
            patches = []
            for _ in range(args.prop_blades):
                poly = Poly3DCollection([np.zeros((4,3))], closed=True,
                                        facecolor="k", edgecolor="none",
                                        alpha=args.prop_alpha)
                ax3d.add_collection3d(poly)
                patches.append(poly)
            blade_patches[li][mj] = patches
            txt = ax3d.text2D(0, 0, "", color="r", fontsize=8,
                              ha="left", va="center", transform=ax3d.transAxes)
            fault_texts[li][mj] = txt

    def link_motor_slice(link_idx):
        j0 = link_idx * motors_per_link
        j1 = j0 + motors_per_link
        return j0, j1

    # right-side bars
    axbars_objs = []
    width = 0.35
    for r, ax in enumerate(axbars):
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Link {r+1} (M1–M{motors_per_link})")
        idxs  = np.arange(motors_per_link)
        x_gt   = idxs - width/2
        x_pred = idxs + width/2

        bars_gt = ax.bar(x_gt, np.zeros(motors_per_link), width=width, alpha=0.35,
                         linewidth=1.0, edgecolor="gray", hatch="//", label="REAL VALUE (fault=1)")
        bars_pd = ax.bar(x_pred, np.zeros(motors_per_link), width=width, label="Pred prob(fault)")

        ax.set_xticks(idxs)
        ax.set_xticklabels([f"M{i+1}" for i in range(motors_per_link)], rotation=0)
        if r == 0: ax.legend(loc="upper right")
        gt_txts = [ax.text(i, 1.02, "", ha="center", va="bottom", fontsize=8) for i in idxs]
        axbars_objs.append((bars_gt, bars_pd, gt_txts))

    legend_lines = [
        plt.Line2D([0],[0], color="k", lw=2.5, label="Actual Link", alpha=0.45),
        plt.Line2D([0],[0], color="g", lw=2.0, linestyle="--", label="Desired Link"),
        plt.Line2D([0],[0], marker="$x$", color="k", lw=0, markersize=8, label="Motor"),
        plt.Line2D([0],[0], marker="$x$", color="r", lw=0, markersize=8, label="Faulty Motor"),
    ]
    ax3d.legend(handles=legend_lines, loc="upper left")

    status_txt = ax3d.text2D(0.58, 0.92, "", transform=ax3d.transAxes, fontsize=14)

    # playback states
    t_idx = [0]
    prob_motor_last  = np.zeros(motors_physical, dtype=float)
    pred_latched_last = 0  # 0=BG, 1..M
    use_multistage = is_multistage
    p_mot_seq_arr = p_mot_seq_np if use_multistage else None
    pred_seq_arr  = pred_seq_np if use_multistage else None

    interval_ms = max(1, int(1000.0 / max(1e-6, args.data_hz * args.speed)))

    def update(_):
        nonlocal pred_latched_last
        t = t_idx[0]
        if t >= T:
            return []

        if use_multistage:
            if p_mot_seq_arr is not None and p_mot_seq_arr.size > 0:
                idx = min(t, p_mot_seq_arr.shape[0]-1)
                prob_motor_last[:] = p_mot_seq_arr[idx]
                pred_latched_last = int(pred_seq_arr[idx])
            else:
                prob_motor_last[:] = 0.0
                pred_latched_last = 0
        else:
            # forward up to t+1 (causal)
            l1, reg, mot = forward_upto(t+1)

            # posterior at *current* time
            p_bg_seq, p_mot_seq = posterior_motor_probs(l1, reg, mot)  # (T,), (T,M)
            p_mot_now= p_mot_seq[-1].detach()

            # latch on whole sequence so far
            preds_seq = latch_motor_kofn_probs(
                p_bg_seq, p_mot_seq,
                theta_m=args.theta_m, kn_k=args.kn_k, kn_nw=args.kn_nw, vote_n=args.vote_n
            )  # (T,)
            pred_latched_last = int(preds_seq[-1].item())  # 0=BG, 1..M
            prob_motor_last[:] = p_mot_now.cpu().numpy()

        # Poses at t
        Td = Dcum[t]; Ta = Acum[t]
        P_d = Td[:, :3, 3]
        P_a = Ta[:, :3, 3]
        if args.fix_origin:
            P_d = P_d - P_d[:1, :]
            P_a = P_a - P_a[:1, :]

        # links
        for i in range(link_count):
            xd, yd, zd = [P_d[i,0], P_d[i+1,0]], [P_d[i,1], P_d[i+1,1]], [P_d[i,2], P_d[i+1,2]]
            desired_lines[i].set_data(xd, yd); desired_lines[i].set_3d_properties(zd)
            xa, ya, za = [P_a[i,0], P_a[i+1,0]], [P_a[i,1], P_a[i+1,1]], [P_a[i,2], P_a[i+1,2]]
            actual_lines[i].set_data(xa, ya);   actual_lines[i].set_3d_properties(za)

        desired_nodes._offsets3d = (P_d[:,0], P_d[:,1], P_d[:,2])
        actual_nodes._offsets3d  = (P_a[:,0], P_a[:,1], P_a[:,2])

        # motor geometry for drawing
        for i in range(link_count):
            R_start = Ta[i,   :3, :3]
            R_end   = Ta[i+1, :3, :3]

            # anchors (앞/뒤 비율 동일)
            ratio = float(args.anchor_ratio)
            p_front = (1.0 - ratio) * P_a[i] + ratio * P_a[i+1]
            p_back  = ratio * P_a[i] + (1.0 - ratio) * P_a[i+1]

            # 8 motors: 4 at front cross, 4 at back cross
            y_end, z_end = R_end[:, 1], R_end[:, 2]
            y_sta, z_sta = R_start[:, 1], R_start[:, 2]
            four_front = np.array([p_front + args.arm_len * y_end,
                                   p_front - args.arm_len * y_end,
                                   p_front + args.arm_len * z_end,
                                   p_front - args.arm_len * z_end])
            four_back  = np.array([p_back  + args.arm_len * y_sta,
                                   p_back  - args.arm_len * y_sta,
                                   p_back  + args.arm_len * z_sta,
                                   p_back  - args.arm_len * z_sta])
            motor_pos  = np.vstack([four_front, four_back])                    # (8,3)

            R_for_blade = [R_end]*4 + [R_start]*4
            anchors     = [p_front]*4 + [p_back]*4

            for j in range(motors_per_link):
                cls1_based = i*motors_per_link + j + 1  # 1..M
                is_fault = (pred_latched_last == cls1_based)
                color_face = "r" if is_fault else "k"

                pj     = motor_pos[j]
                p_anc  = anchors[j]
                R_ref  = R_for_blade[j]

                # stem
                stems_lines[i][j].set_data([p_anc[0], pj[0]], [p_anc[1], pj[1]])
                stems_lines[i][j].set_3d_properties([p_anc[2], pj[2]])
                stems_lines[i][j].set_color(color_face)

                # blades
                n_hat = _norm(pj - p_anc)
                u_ref = R_ref[:,1]; v_ref = R_ref[:,2]
                u = u_ref - np.dot(u_ref, n_hat)*n_hat
                if np.linalg.norm(u) < 1e-6:
                    u = v_ref - np.dot(v_ref, n_hat)*n_hat
                u = u / (np.linalg.norm(u)+1e-9)
                v = np.cross(n_hat, u); v = v/(np.linalg.norm(v)+1e-9)

                # alternating spin direction (if enabled)
                spin_sign = 1.0 if (args.spin_dir_alt==0 or (j%2==0)) else -1.0
                base_phase = 0.0 if is_fault else spin_sign * 2.0*np.pi*args.prop_rps * (1.0/args.data_hz) * t

                for k, poly in enumerate(blade_patches[i][j]):
                    theta = base_phase + 2.0*np.pi * (k / max(1, args.prop_blades))
                    c = np.cos(theta); s = np.sin(theta)
                    axis =  c * u + s * v
                    perp = -s * u + c * v
                    r_root = 0.25 * args.prop_radius; r_tip  = args.prop_radius
                    half_c = 0.5  * args.prop_chord
                    root = pj + r_root * axis
                    tip  = pj + r_tip  * axis
                    p1 = root + half_c * perp; p2 = tip + half_c * perp
                    p3 = tip - half_c * perp;  p4 = root - half_c * perp
                    quad  = np.stack([p1, p2, p3, p4], axis=0)
                    poly.set_verts([quad])
                    poly.set_facecolor(color_face)
                    poly.set_edgecolor("none")
                    poly.set_alpha(args.prop_alpha)

                # label for faulted motor
                label = fault_texts[i][j]
                if is_fault:
                    x2, y2, _ = proj3d.proj_transform(pj[0], pj[1], pj[2], ax3d.get_proj())
                    label.set_text(f"Link{i+1} M{j+1} Fault")
                    label.set_position((x2 + 0.005, y2 + 0.005))
                    label.set_color("r"); label.set_alpha(1.0)
                    label.set_visible(True)
                else:
                    label.set_visible(False)

        # right bars
        gt_vec = Y_with_bg[t]               # (1+M)
        gt_motor = gt_vec[1:]               # (M,)
        for r in range(rows):
            j0 = r * motors_per_link
            j1 = j0 + motors_per_link
            bars_gt, bars_pred, gt_txts = axbars_objs[r]
            for i_m in range(motors_per_link):
                m_idx = j0 + i_m
                bars_gt[i_m].set_height(float(gt_motor[m_idx]))
                bars_pred[i_m].set_height(float(prob_motor_last[m_idx]))
                is_alarm = (pred_latched_last == (m_idx+1))  # 1-based
                bars_pred[i_m].set_edgecolor("r" if is_alarm else "black")
                bars_pred[i_m].set_linewidth(2.5 if is_alarm else 0.5)
                gt_txts[i_m].set_text("GT:F" if gt_motor[m_idx] >= 0.5 else "")

        t_real = t / max(1e-6, args.data_hz)
        status_txt.set_text(f"t = {t_real:4.2f}s")

        t_idx[0] += 1

        artists = desired_lines + actual_lines + [desired_nodes, actual_nodes, base_marker, base_text, status_txt]
        for i in range(link_count):
            artists.extend(stems_lines[i])
            for patches in blade_patches[i]:
                artists.extend(patches)
        for r in range(rows):
            bars_gt, bars_pred, gt_txts = axbars_objs[r]
            artists.extend(list(bars_gt))
            artists.extend(list(bars_pred))
            artists.extend(gt_txts)
        return artists

    # animation
    interval_ms = max(1, int(1000.0 / max(1e-6, args.data_hz * args.speed)))
    ani = FuncAnimation(
        fig, update,
        interval=interval_ms,
        blit=False,
        save_count=T,
        cache_frame_data=False
    )
    plt.tight_layout()

    # save or show
    def _parse_bitrate_to_kbps(b):
        s = str(b).strip().lower()
        if s.endswith("k"): s = s[:-1]
        return int(float(s))

    if args.save_video:
        ext = os.path.splitext(args.out)[1].lower()
        try:
            if ext in [".mp4", ".m4v", ".mov"]:
                writer = FFMpegWriter(fps=args.video_fps, codec=args.codec,
                                      bitrate=_parse_bitrate_to_kbps(args.bitrate))
            elif ext in [".gif"]:
                writer = PillowWriter(fps=args.video_fps)
            else:
                raise ValueError(f"Unsupported extension: {ext} (use .mp4 or .gif)")
            print(f"💾 Saving video to: {args.out}  (fps={args.video_fps}, dpi={args.dpi})")
            ani.save(args.out, writer=writer, dpi=args.dpi)
            print("✅ Done.")
        except Exception as e:
            print(f"❌ Video save failed: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    main()




"""
디버깅/실행 예시 (훈련 체크포인트/데이터에 맞춰 경로 조정):
python3 causal_tcn/visualize_stream_lstm_fdi.py \
  --ckpt FDI_MultiStage/link_3/FDI_MultiStage_ALL.pth \
  --npz  data_storage/link_3/fault_dataset.npz \
  --seq_idx 1234 \
  --dt 0.01 --data_hz 100 --speed 1.0 \
  --label_fault_is_one 0 \
  --motors_per_link 8

"""



""" MP4저장
python3 causal_tcn/visualize_stream_causal_tcn.py \
  --ckpt TCN/TCN_link_1_RELonly_CAUSAL.pth \
  --npz  data_storage/link_1/fault_dataset.npz \
  --seq_idx 100 \
  --threshold 0.5 --kofn 3,5 \
  --data_hz 100 --speed 3.33 \
  --prepend_base 1 --fix_origin 1 \
  --label_fault_is_one 1 \
  --motors_per_link 8 \
  --anchor_ratio 0.15 \
  --arm_len 0.15 \
  --prop_blades 4 --prop_radius 0.08 --prop_chord 0.028 --prop_alpha 0.85 \
  --save_video 1 \
  --out data_storage/link_1/vis.mp4 \
  --video_fps 30 --codec libx264 --bitrate 6000k --dpi 150
"""

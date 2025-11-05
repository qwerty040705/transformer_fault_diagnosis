
# fault_detect/features.py
from __future__ import annotations
import torch
import numpy as np
from typing import Tuple

# ----- SO(3) / SE(3) helpers (torch, batched) -----

def _vee(mat: torch.Tensor) -> torch.Tensor:
    # mat: (...,3,3) skew-symmetric
    return torch.stack([
        mat[..., 2, 1] - mat[..., 1, 2],
        mat[..., 0, 2] - mat[..., 2, 0],
        mat[..., 1, 0] - mat[..., 0, 1]
    ], dim=-1)

def so3_log(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    R: (...,3,3) rotation matrices
    return: (...,3) rotation vector (axis * angle)
    """
    # clamp trace into [-1,3] numerically safe
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)

    # For small angles, use first-order approx: log(R) ~ 0.5*(R - R^T)
    S = 0.5 * (R - R.transpose(-1, -2))
    v = _vee(S)

    small = (theta < 1e-3).unsqueeze(-1)  # (...,1)
    sin_theta = torch.sin(theta).unsqueeze(-1)

    # General: v * theta / sin(theta)
    scale = theta.unsqueeze(-1) / (sin_theta + eps)
    v_general = v * scale

    return torch.where(small, v, v_general)

def se3_err(T_act: torch.Tensor, T_des: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    T_act, T_des: (...,4,4)
    Returns:
      e_r: (...,3) orientation error (rotvec of R_des^T R_act)
      e_p: (...,3) position error (p_act - p_des)
    """
    R_act = T_act[..., :3, :3]
    R_des = T_des[..., :3, :3]
    p_act = T_act[..., :3, 3]
    p_des = T_des[..., :3, 3]
    R_err = torch.matmul(R_des.transpose(-1, -2), R_act)  # R_des^T R_act
    e_r = so3_log(R_err)
    e_p = p_act - p_des
    return e_r, e_p

def diff_along_time(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    x: (T, ...), finite difference derivative (prepend zero)
    """
    dx = (x[1:] - x[:-1]) / dt
    zero = torch.zeros_like(dx[:1])
    return torch.cat([zero, dx], dim=0)

# ----- Feature builder -----

@torch.no_grad()
def build_link_features(
    desired_link_rel: np.ndarray, actual_link_rel: np.ndarray,
    desired_link_cum: np.ndarray, actual_link_cum: np.ndarray,
    desired_ee: np.ndarray, actual_ee: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Inputs:
      *_link_rel/cum: (T, L, 4, 4)
      *_ee: (T, 4, 4)
    Returns:
      features: (T, L, F) with F=36 by default
        [rel(rot 3 + pos 3) + cum(rot 3 + pos 3) + ee(rot 3 + pos 3)] + time-derivatives (x2) = 36
    """
    device = torch.device("cpu")


    T, L = desired_link_rel.shape[:2]

    def to_torch(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    Td_rel = to_torch(desired_link_rel)   # (T,L,4,4)
    Ta_rel = to_torch(actual_link_rel)
    Td_cum = to_torch(desired_link_cum)
    Ta_cum = to_torch(actual_link_cum)
    Td_ee  = to_torch(desired_ee)         # (T,4,4)
    Ta_ee  = to_torch(actual_ee)

    # Expand EE to per-link for simple concatenation
    Td_ee_L = Td_ee.unsqueeze(1).expand(T, L, 4, 4).contiguous()
    Ta_ee_L = Ta_ee.unsqueeze(1).expand(T, L, 4, 4).contiguous()

    # Compute per (T,L)
    e_rr, e_rp = se3_err(Ta_rel, Td_rel)  # (T,L,3), (T,L,3)
    e_cr, e_cp = se3_err(Ta_cum, Td_cum)
    e_er, e_ep = se3_err(Ta_ee_L, Td_ee_L)

    # Stack base features
    base = torch.cat([e_rr, e_rp, e_cr, e_cp, e_er, e_ep], dim=-1)  # (T,L,18)

    # Derivatives along time
    d_rr = diff_along_time(e_rr, dt)
    d_rp = diff_along_time(e_rp, dt)
    d_cr = diff_along_time(e_cr, dt)
    d_cp = diff_along_time(e_cp, dt)
    d_er = diff_along_time(e_er, dt)
    d_ep = diff_along_time(e_ep, dt)
    d_feats = torch.cat([d_rr, d_rp, d_cr, d_cp, d_er, d_ep], dim=-1)  # (T,L,18)

    feats = torch.cat([base, d_feats], dim=-1)  # (T,L,36)
    return feats.cpu().numpy()

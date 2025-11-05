# fault_detect/utils.py
from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def compute_timewise_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    y_true: (N,8) 0/1 fault targets
    y_prob: (N,8) probabilities
    """
    y_pred = (y_prob >= threshold).astype(np.uint8)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    y_prob_flat = y_prob.reshape(-1)

    try:
        auroc = roc_auc_score(y_true_flat, y_prob_flat)
    except Exception:
        auroc = float("nan")
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    bacc = balanced_accuracy_score(y_true_flat, y_pred_flat)
    return {"auroc": auroc, "f1": f1, "bacc": bacc}

def onset_mae(true_onset: np.ndarray, pred_onset: np.ndarray, dt: float) -> float:
    """
    onset arrays: (8,)
    """
    valid = (true_onset >= 0) & (pred_onset >= 0)
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs((true_onset[valid] - pred_onset[valid]) * dt)))

# fault_detect/postprocess.py
from __future__ import annotations
import numpy as np

def ewma(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    x: (T, 8) probabilities
    """
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, x.shape[0]):
        y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
    return y

def hysteresis(p: np.ndarray, th_on: float = 0.6, th_off: float = 0.4) -> np.ndarray:
    """
    p: (T,8) probs -> binary decisions (T,8)
    """
    T, M = p.shape
    y = np.zeros((T, M), dtype=np.uint8)
    state = np.zeros((M,), dtype=np.uint8)
    for t in range(T):
        for m in range(M):
            if state[m]:
                if p[t, m] < th_off:
                    state[m] = 0
            else:
                if p[t, m] >= th_on:
                    state[m] = 1
        y[t] = state
    return y

def first_onset(decisions: np.ndarray) -> np.ndarray:
    """
    decisions: (T,8) 0/1 -> onset index per motor, -1 if none
    """
    T, M = decisions.shape
    onset = -np.ones((M,), dtype=np.int32)
    for m in range(M):
        idx = np.where(decisions[:, m] == 1)[0]
        if idx.size > 0:
            onset[m] = int(idx[0])
    return onset

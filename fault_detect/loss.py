# fault_detect/loss.py
from __future__ import annotations
import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits, target: (B,T,8)
        # BCEWithLogitsLoss는 브로드캐스트 허용. 채널별 동일 가중.
        return self.bce(logits, target)

def onset_weighted_bce(logits: torch.Tensor, target: torch.Tensor, onset_mask: torch.Tensor, weight: float = 3.0):
    """
    onset 주변(마스크=1)에 가중치 부여
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none'
    )  # (B,T,8)
    w = 1.0 + (weight - 1.0) * onset_mask  # (B,T,8)
    return (bce * w).mean()

def total_variation_loss(probs: torch.Tensor, lam: float = 0.05):
    """
    probs: (B,T,8) after sigmoid
    Encourages temporal smoothness
    """
    diff = probs[:, 1:, :] - probs[:, :-1, :]
    return lam * diff.abs().mean()

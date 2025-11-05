# fault_detect/model.py
from __future__ import annotations
import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, dropout=0.1):
        super().__init__()
        k = 3
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # (B,C,T)
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        return y + self.res(x)

class LinkTemporalModel(nn.Module):
    """
    입력: (B, T, F)  -> 출력: (B, T, 8)
    """
    def __init__(self, in_dim=36, hidden=192, nheads=8, nlayers=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.tcn = nn.Sequential(
            TCNBlock(hidden, hidden, dilation=1, dropout=dropout),
            TCNBlock(hidden, hidden, dilation=2, dropout=dropout),
            TCNBlock(hidden, hidden, dilation=4, dropout=dropout),
            TCNBlock(hidden, hidden, dilation=8, dropout=dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nheads, batch_first=True,
            dim_feedforward=hidden*4, dropout=dropout, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 8)
        )

    def forward(self, x):  # x: (B,T,F)
        h = self.in_proj(x)               # (B,T,H)
        h = h.transpose(1, 2)             # (B,H,T)
        h = self.tcn(h)                   # (B,H,T)
        h = h.transpose(1, 2)             # (B,T,H)
        h = self.encoder(h)               # (B,T,H)
        logits = self.head(h)             # (B,T,8)
        return logits

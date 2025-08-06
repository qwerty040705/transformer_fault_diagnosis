import math
import torch
import torch.nn as nn

class FaultDiagnosisTransformer(nn.Module):
    def __init__(
        self,
        input_dim=24,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_dim=8,
        max_seq_len=200,
        mlp_head=True,            # 분류 헤드를 MLP로 할지 여부
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        # [B, 1, d], [1, T+1, d]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        self.pos_drop = nn.Dropout(p=dropout)

        # batch_first=True 로 경고 제거 + 성능 향상
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),  # 최종 LayerNorm
        )

        if mlp_head:
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim),
            )
        else:
            self.head = nn.Linear(d_model, output_dim)

        # init
        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):  # x: [B, T, input_dim]
        B, T, _ = x.shape
        if T > self.max_seq_len:
            raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}. "
                             f"Increase max_seq_len when constructing the model.")

        z = self.input_proj(x)                   # [B, T, d]
        z = z / math.sqrt(self.d_model)          # embedding scaling

        cls = self.cls_token.expand(B, 1, -1)    # [B, 1, d]
        z = torch.cat([cls, z], dim=1)           # [B, T+1, d]
        z = z + self.pos_embedding[:, :T+1, :]   # [B, T+1, d]
        z = self.pos_drop(z)

        z = self.encoder(z)                      # [B, T+1, d]
        cls_out = z[:, 0, :]                     # [B, d]
        logits = self.head(cls_out)              # [B, output_dim]
        return logits

import torch
import torch.nn as nn

class FaultDiagnosisTransformer(nn.Module):
    def __init__(self, input_dim=24, d_model=64, nhead=8, num_layers=2,
                 dim_feedforward=128, dropout=0.1, output_dim=8, max_seq_len=200):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # learnable positional embedding (CLS 포함 → T+1)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 구버전 호환: batch_first 없이 사용 (입력: [S, B, E])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

        # init
        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):  # x: [B, T, 24]
        B, T, _ = x.shape
        z = self.input_proj(x)                 # [B, T, d]
        cls = self.cls_token.expand(B, 1, -1)  # [B, 1, d]
        z = torch.cat([cls, z], dim=1)         # [B, T+1, d]
        z = z + self.pos_embedding[:, :T+1, :] # [B, T+1, d]

        # Transformer는 [S, B, E] 포맷
        z = z.transpose(0, 1)                  # [T+1, B, d]
        z = self.encoder(z)                    # [T+1, B, d]
        z = z.transpose(0, 1)                  # [B, T+1, d]

        cls_out = z[:, 0, :]                   # [B, d]
        logits = self.head(cls_out)            # [B, M]
        return logits

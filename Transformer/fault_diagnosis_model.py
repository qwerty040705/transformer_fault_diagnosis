import math
import torch
import torch.nn as nn

class FaultDiagnosisTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 24,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: int = 8,
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding for T timesteps (no cls token)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.pos_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),  # Output per timestep
        )

        # Initialization
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_embedding, std=0.02)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            out: (B, T, output_dim)
        """
        B, T, _ = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"T={T} exceeds max_seq_len={self.max_seq_len}. "
                f"Increase max_seq_len when constructing the model."
            )

        z = self.input_proj(x) / math.sqrt(self.d_model)  # (B, T, d_model)
        z = z + self.pos_embedding[:, :T, :]              # (B, T, d_model)
        z = self.pos_drop(z)
        z = self.encoder(z)                               # (B, T, d_model)
        return self.head(z)                               # (B, T, output_dim)

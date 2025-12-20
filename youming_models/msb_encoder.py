import torch
from torch import nn


class MSBEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_patches: int = 1024,  # 256x256 / 8x8
    ):
        super().__init__()
        ff_dim = int(mlp_ratio * embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        if N == self.pos_embed.shape[1]:
            x = tokens + self.pos_embed
        else:
            # Interpolate pos_embed
            # Assuming square grid for simplicity, or just 1D interp
            # 1D interp is safer if we don't know grid shape here
            pos_embed = nn.functional.interpolate(
                self.pos_embed.transpose(1, 2), size=N, mode="linear", align_corners=False
            ).transpose(1, 2)
            x = tokens + pos_embed
        return self.encoder(x)

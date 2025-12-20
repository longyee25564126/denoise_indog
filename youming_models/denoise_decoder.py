import torch
from torch import nn


def unpatchify(patches: torch.Tensor, h: int, w: int, patch_size: int) -> torch.Tensor:
    """
    Convert patch tokens of shape (B, N, 3*P*P) back to an image (B, 3, H, W)
    where N = h*w, H = h*P, W = w*P.
    """
    #111
    B, N, PP = patches.shape
    P = patch_size
    assert PP == 3 * P * P, f"Expected channel size 3*P*P, got {PP}"
    assert N == h * w, f"N ({N}) != h*w ({h*w})"

    patches = patches.view(B, h, w, 3, P, P)
    img = (
        patches.permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(B, 3, h * P, w * P)
    )
    return img


class DenoiseDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        patch_size: int = 8,
        use_concat_fuse: bool = False,  # Deprecated
        clamp_output: bool = False,
        max_patches: int = 1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.clamp_output = clamp_output

        # Cross Attention: Query=LSB, Key=MSB, Value=MSB
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        ff_dim = int(mlp_ratio * embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3 * patch_size * patch_size),
        )
        self.output_scale = nn.Parameter(torch.tensor(10.0))

    def forward(
        self,
        x_rgb: torch.Tensor,
        msb_tokens: torch.Tensor,
        lsb_tokens: torch.Tensor,
        m_hat_tok: torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> dict:
        h, w = grid_shape
        B, N, D = msb_tokens.shape
        assert lsb_tokens.shape == (B, N, D)
        assert m_hat_tok.shape[0] == B and m_hat_tok.shape[1] == N

        # Cross Attention: Q=LSB, K=MSB, V=MSB
        # T_in = Norm(LSB + CrossAttn(LSB, MSB, MSB))
        attn_out, _ = self.cross_attn(query=lsb_tokens, key=msb_tokens, value=msb_tokens)
        T_in = self.norm1(lsb_tokens + self.dropout1(attn_out))

        if N == self.pos_embed.shape[1]:
            T_in = T_in + self.pos_embed
        else:
            pos_embed = nn.functional.interpolate(
                self.pos_embed.transpose(1, 2), size=N, mode="linear", align_corners=False
            ).transpose(1, 2)
            T_in = T_in + pos_embed

        T_dec = self.decoder(T_in)  # (B, N, D)

        R_tok = self.residual_head(T_dec) * self.output_scale  # (B, N, 3*P*P)
        # R_tok_gated = R_tok * m_hat_tok  # Removed gating to avoid vanishing gradient
        
        residual_gated = unpatchify(R_tok, h, w, self.patch_size)  # (B,3,H,W)

        y_hat = x_rgb - residual_gated
        if self.clamp_output:
            y_hat = y_hat.clamp(0.0, 1.0)

        return {"residual_gated": residual_gated, "y_hat": y_hat}

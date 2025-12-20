import torch
from torch import nn


def unpatchify(patches: torch.Tensor, h: int, w: int, patch_size: int) -> torch.Tensor:
    """
    Convert patch tokens of shape (B, N, 3*P*P) back to an image (B, 3, H, W)
    where N = h*w, H = h*P, W = w*P.
    """
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
        use_concat_fuse: bool = True,
        clamp_output: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.clamp_output = clamp_output

        self.fuse = nn.Linear(2 * embed_dim, embed_dim) if use_concat_fuse else nn.Identity()

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

        self.residual_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3 * patch_size * patch_size),
        )

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

        T_cat = torch.cat([msb_tokens, lsb_tokens], dim=-1)  # (B, N, 2D)
        T_in = self.fuse(T_cat)  # (B, N, D)

        T_dec = self.decoder(T_in)  # (B, N, D)

        R_tok = self.residual_head(T_dec)  # (B, N, 3*P*P)
        R_tok_gated = R_tok * m_hat_tok  # broadcast gating

        residual_gated = unpatchify(R_tok_gated, h, w, self.patch_size)  # (B,3,H,W)

        y_hat = x_rgb - residual_gated
        if self.clamp_output:
            y_hat = y_hat.clamp(0.0, 1.0)

        return {"residual_gated": residual_gated, "y_hat": y_hat}


class DenoiseDecoderQMSB(nn.Module):
    """
    TransformerDecoder style: queries from MSB tokens attend to LSB tokens (memory).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        patch_size: int = 8,
        clamp_output: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.clamp_output = clamp_output

        ff_dim = int(mlp_ratio * embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)

        self.residual_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3 * patch_size * patch_size),
        )

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

        T_dec = self.decoder(tgt=msb_tokens, memory=lsb_tokens)  # (B, N, D)

        R_tok = self.residual_head(T_dec)  # (B, N, 3*P*P)
        R_tok_gated = R_tok * m_hat_tok  # broadcast gating

        residual_gated = unpatchify(R_tok_gated, h, w, self.patch_size)  # (B,3,H,W)

        y_hat = x_rgb - residual_gated
        if self.clamp_output:
            y_hat = y_hat.clamp(0.0, 1.0)

        return {"residual_gated": residual_gated, "y_hat": y_hat}


class DenoiseDecoderStdEncDec(nn.Module):
    """
    Standard Transformer encoder-decoder style:
    - memory: encoded MSB tokens
    - tgt/queries: learnable 2D query grid interpolated to current patch grid (no LSB / mask gating)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        patch_size: int = 8,
        query_grid_size: tuple[int, int] = (16, 16),
        clamp_output: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.clamp_output = clamp_output

        ff_dim = int(mlp_ratio * embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)

        self.query_embed_2d = nn.Parameter(
            torch.randn(1, embed_dim, query_grid_size[0], query_grid_size[1])
        )

        self.residual_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3 * patch_size * patch_size),
        )

    def forward(
        self,
        x_rgb: torch.Tensor,
        msb_tokens: torch.Tensor,
        lsb_tokens: torch.Tensor | None,
        m_hat_tok: torch.Tensor | None,
        grid_shape: tuple[int, int],
    ) -> dict:
        h, w = grid_shape
        B, N, D = msb_tokens.shape
        assert N == h * w, f"Token count {N} != grid {h}x{w}"

        # Interpolate learnable 2D queries to current grid then flatten
        q2d = torch.nn.functional.interpolate(
            self.query_embed_2d,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )  # (1, D, h, w)
        q = q2d.flatten(2).permute(0, 2, 1)  # (1, h*w, D)
        q = q.expand(B, -1, -1).contiguous()  # (B, N, D)

        T_dec = self.decoder(tgt=q, memory=msb_tokens)  # (B, N, D)

        R_tok = self.residual_head(T_dec)  # (B, N, 3*P*P)
        residual = unpatchify(R_tok, h, w, self.patch_size)  # (B,3,H,W)

        y_hat = x_rgb - residual
        if self.clamp_output:
            y_hat = y_hat.clamp(0.0, 1.0)

        return {"residual_gated": residual, "y_hat": y_hat}

import torch
from torch import nn

from .tokenizer_cnn import ConvPatchTokenizer
from .msb_encoder import MSBEncoder
from .lsb_mask_head import LSBMaskHead
from .denoise_decoder import DenoiseDecoder


class BitPlaneFormerV1(nn.Module):
    def __init__(
        self,
        patch_size: int = 8,
        embed_dim: int = 256,
        num_heads: int = 8,
        msb_depth: int = 6,
        dec_depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.msb_tokenizer = ConvPatchTokenizer(in_ch=12, embed_dim=embed_dim, patch_size=patch_size)
        self.lsb_tokenizer = ConvPatchTokenizer(in_ch=12, embed_dim=embed_dim, patch_size=patch_size)
        self.msb_encoder = MSBEncoder(embed_dim=embed_dim, num_heads=num_heads, depth=msb_depth, mlp_ratio=mlp_ratio, dropout=dropout)
        self.mask_head = LSBMaskHead(embed_dim=embed_dim, hidden_dim=None, dropout=dropout)
        self.decoder = DenoiseDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=dec_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            patch_size=patch_size,
            use_concat_fuse=True,
            clamp_output=False,
        )

    def forward(self, x: torch.Tensor | dict, lsb: torch.Tensor | None = None, msb: torch.Tensor | None = None) -> dict:
        """
        Forward either with explicit tensors (x, lsb, msb) or a dict containing them.
        """
        if isinstance(x, dict):
            batch = x
            x_rgb = batch["x"]
            lsb = batch["lsb"]
            msb = batch["msb"]
        else:
            assert lsb is not None and msb is not None, "lsb and msb must be provided"
            x_rgb = x

        T_msb0, grid_shape = self.msb_tokenizer(msb)
        T_lsb0, grid_shape_lsb = self.lsb_tokenizer(lsb)
        assert grid_shape == grid_shape_lsb, "MSB/LSB grids mismatch"

        T_msb = self.msb_encoder(T_msb0)
        mask_out = self.mask_head(T_lsb0, grid_shape)
        dec_out = self.decoder(x_rgb, T_msb, T_lsb0, mask_out["m_hat_tok"], grid_shape)

        out = {
            "y_hat": dec_out["y_hat"],
            "residual_gated": dec_out["residual_gated"],
            "m_logits": mask_out["m_logits"],
            "m_hat": mask_out["m_hat"],
        }
        return out

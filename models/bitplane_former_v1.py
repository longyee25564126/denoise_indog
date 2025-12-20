import torch
from torch import nn

from .tokenizer_cnn import ConvPatchTokenizer
from .msb_encoder import MSBEncoder
from .lsb_mask_head import LSBMaskHead
from .denoise_decoder import DenoiseDecoder
from datasets.bitplane_utils import expand_bits


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
        lsb_bits: tuple[int, int] | list[int] = (0, 5),
        msb_bits: tuple[int, int] | list[int] = (6, 7),
    ):
        super().__init__()
        self.patch_size = patch_size

        lsb_bit_list = expand_bits(lsb_bits)
        msb_bit_list = expand_bits(msb_bits)
        self.lsb_in_ch = 3 * len(lsb_bit_list)
        self.msb_in_ch = 3 * len(msb_bit_list)

        self.msb_tokenizer = ConvPatchTokenizer(in_ch=self.msb_in_ch, embed_dim=embed_dim, patch_size=patch_size)
        self.lsb_tokenizer = ConvPatchTokenizer(in_ch=self.lsb_in_ch, embed_dim=embed_dim, patch_size=patch_size)
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

    def forward(
        self,
        x: torch.Tensor | dict,
        lsb: torch.Tensor | None = None,
        msb: torch.Tensor | None = None,
        mask_override: torch.Tensor | None = None,
    ) -> dict:
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

        # Sanity checks on channel counts
        assert lsb.shape[1] == self.lsb_in_ch, f"Expected LSB channels {self.lsb_in_ch}, got {lsb.shape[1]}"
        assert msb.shape[1] == self.msb_in_ch, f"Expected MSB channels {self.msb_in_ch}, got {msb.shape[1]}"

        T_msb0, grid_shape = self.msb_tokenizer(msb)
        T_lsb0, grid_shape_lsb = self.lsb_tokenizer(lsb)
        assert grid_shape == grid_shape_lsb, "MSB/LSB grids mismatch"

        T_msb = self.msb_encoder(T_msb0)
        mask_out = self.mask_head(T_lsb0, grid_shape)
        m_hat = mask_out["m_hat"]
        m_hat_tok = mask_out["m_hat_tok"]

        if mask_override is not None:
            if mask_override.ndim == 3:
                mask_override = mask_override.unsqueeze(0)
            assert mask_override.shape[0] == x_rgb.shape[0], "mask_override batch mismatch"
            h, w = grid_shape
            assert mask_override.shape[2] == h and mask_override.shape[3] == w, "mask_override spatial mismatch"
            m_hat = mask_override
            m_hat_tok = m_hat.permute(0, 2, 3, 1).contiguous().view(x_rgb.shape[0], -1, 1)

        dec_out = self.decoder(x_rgb, T_msb, T_lsb0, m_hat_tok, grid_shape)

        out = {
            "y_hat": dec_out["y_hat"],
            "residual_gated": dec_out["residual_gated"],
            "m_logits": mask_out["m_logits"],
            "m_hat": m_hat,
        }
        return out

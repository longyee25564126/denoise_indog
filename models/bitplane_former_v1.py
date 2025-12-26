import torch
import torch.nn.functional as F
from torch import nn

from .tokenizer_cnn import ConvPatchTokenizer
from .msb_encoder import MSBEncoder
from .lsb_mask_head import LSBMaskHead
from .denoise_decoder import DenoiseDecoder, DenoiseDecoderQMSB, DenoiseDecoderStdEncDec
from datasets.bitplane_utils import expand_bits


class BitPlaneFormerV1(nn.Module):
    def __init__(
        self,
        patch_size: int = 8,
        patch_stride: int | None = None,
        pad_size: int = 0,
        embed_dim: int = 256,
        num_heads: int = 8,
        msb_depth: int = 6,
        dec_depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        lsb_bits: tuple[int, int] | list[int] = (0, 5),
        msb_bits: tuple[int, int] | list[int] = (6, 7),
        dec_type: str = "fuse_encoder",
        lsb_depth: int | None = None,
        fusion_type: str = "concat",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        if pad_size < 0:
            raise ValueError("pad_size must be >= 0")
        if pad_size % self.patch_stride != 0:
            raise ValueError("pad_size must be divisible by patch_stride")
        self.pad_size = pad_size

        lsb_bit_list = expand_bits(lsb_bits)
        msb_bit_list = expand_bits(msb_bits)
        self.lsb_in_ch = 3 * len(lsb_bit_list)
        self.msb_in_ch = 3 * len(msb_bit_list)

        self.msb_tokenizer = ConvPatchTokenizer(
            in_ch=self.msb_in_ch,
            embed_dim=embed_dim,
            patch_size=patch_size,
            patch_stride=self.patch_stride,
        )
        self.lsb_tokenizer = ConvPatchTokenizer(
            in_ch=self.lsb_in_ch,
            embed_dim=embed_dim,
            patch_size=patch_size,
            patch_stride=self.patch_stride,
        )
        self.msb_encoder = MSBEncoder(embed_dim=embed_dim, num_heads=num_heads, depth=msb_depth, mlp_ratio=mlp_ratio, dropout=dropout)
        if lsb_depth is None:
            lsb_depth = msb_depth
        self.lsb_encoder = MSBEncoder(embed_dim=embed_dim, num_heads=num_heads, depth=lsb_depth, mlp_ratio=mlp_ratio, dropout=dropout)
        self.mask_head = LSBMaskHead(embed_dim=embed_dim, hidden_dim=None, dropout=dropout)
        dec_type = dec_type.lower()
        if dec_type == "fuse_encoder":
            self.decoder = DenoiseDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=dec_depth,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                patch_size=patch_size,
                patch_stride=self.patch_stride,
                fusion_type=fusion_type,
                clamp_output=False,
            )
        elif dec_type == "decoder_q_msb":
            self.decoder = DenoiseDecoderQMSB(
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=dec_depth,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                patch_size=patch_size,
                patch_stride=self.patch_stride,
                clamp_output=False,
            )
        elif dec_type == "std_encdec_msb":
            self.decoder = DenoiseDecoderStdEncDec(
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=dec_depth,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                patch_size=patch_size,
                patch_stride=self.patch_stride,
                clamp_output=False,
            )
        else:
            raise ValueError(f"Unknown dec_type {dec_type}")
        self.dec_type = dec_type

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
        assert msb.shape[1] == self.msb_in_ch, f"Expected MSB channels {self.msb_in_ch}, got {msb.shape[1]}"
        assert lsb is not None and lsb.shape[1] == self.lsb_in_ch, f"Expected LSB channels {self.lsb_in_ch}, got {lsb.shape[1]}"

        # Optional zero padding for better border context
        if self.pad_size > 0:
            pad = self.pad_size
            x_rgb = F.pad(x_rgb, (pad, pad, pad, pad), mode="constant", value=0.0)
            msb = F.pad(msb, (pad, pad, pad, pad), mode="constant", value=0.0)
            lsb = F.pad(lsb, (pad, pad, pad, pad), mode="constant", value=0.0)

        # Tokenize
        T_msb0, grid_shape = self.msb_tokenizer(msb)
        T_msb = self.msb_encoder(T_msb0)

        T_lsb0, grid_shape_lsb = self.lsb_tokenizer(lsb)
        assert grid_shape == grid_shape_lsb, "MSB/LSB grids mismatch"
        T_lsb = self.lsb_encoder(T_lsb0)

        mask_out = self.mask_head(T_lsb, grid_shape)
        m_logits_full = mask_out["m_logits"]
        m_hat_full = mask_out["m_hat"]
        m_hat_tok_full = mask_out["m_hat_tok"]

        if mask_override is not None:
            if mask_override.ndim == 3:
                mask_override = mask_override.unsqueeze(0)
            assert mask_override.shape[0] == x_rgb.shape[0], "mask_override batch mismatch"
            h, w = grid_shape
            pad_tokens = self.pad_size // self.patch_stride
            if pad_tokens > 0:
                exp_h = h - 2 * pad_tokens
                exp_w = w - 2 * pad_tokens
                assert mask_override.shape[2] == exp_h and mask_override.shape[3] == exp_w, "mask_override spatial mismatch"
                mask_override_pad = F.pad(
                    mask_override,
                    (pad_tokens, pad_tokens, pad_tokens, pad_tokens),
                    mode="constant",
                    value=0.0,
                )
                assert mask_override_pad.shape[2] == h and mask_override_pad.shape[3] == w, "mask_override spatial mismatch"
                m_hat_full = mask_override_pad
            else:
                assert mask_override.shape[2] == h and mask_override.shape[3] == w, "mask_override spatial mismatch"
                m_hat_full = mask_override
            m_hat_tok_full = m_hat_full.permute(0, 2, 3, 1).contiguous().view(x_rgb.shape[0], -1, 1)

        dec_out = self.decoder(x_rgb, T_msb, T_lsb, m_hat_tok_full, grid_shape)

        if self.pad_size > 0:
            pad = self.pad_size
            dec_out["residual_gated"] = dec_out["residual_gated"][..., pad:-pad, pad:-pad]
            dec_out["y_hat"] = dec_out["y_hat"][..., pad:-pad, pad:-pad]

        if self.pad_size > 0:
            pad_tokens = self.pad_size // self.patch_stride
            if pad_tokens > 0:
                m_hat = m_hat_full[:, :, pad_tokens:-pad_tokens, pad_tokens:-pad_tokens]
                m_logits = m_logits_full[:, :, pad_tokens:-pad_tokens, pad_tokens:-pad_tokens]
            else:
                m_hat = m_hat_full
                m_logits = m_logits_full
        else:
            m_hat = m_hat_full
            m_logits = m_logits_full

        out = {
            "y_hat": dec_out["y_hat"],
            "residual_gated": dec_out["residual_gated"],
            "m_logits": m_logits,
            "m_hat": m_hat,
        }
        return out

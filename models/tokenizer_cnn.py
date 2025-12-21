import torch
from torch import nn


class ConvPatchTokenizer(nn.Module):
    """
    Simple conv-based patch tokenizer: (B, C, H, W) -> (B, N, D) tokens.
    """

    def __init__(
        self,
        in_ch: int,
        embed_dim: int = 256,
        patch_size: int = 8,
        patch_stride: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        if patch_stride is None:
            patch_stride = patch_size
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if patch_stride > patch_size:
            raise ValueError("patch_stride should not exceed patch_size (would create gaps)")
        self.patch_stride = patch_stride
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=bias)

    def forward(self, x_planes: torch.Tensor):
        B, C, H, W = x_planes.shape
        P = self.patch_size
        S = self.patch_stride
        assert C == self.in_ch, f"Expected {self.in_ch} channels, got {C}"
        assert H >= P and W >= P, "Input spatial size must be >= patch_size"
        assert (H - P) % S == 0 and (W - P) % S == 0, (
            f"Input spatial size must satisfy (H-P)%S==0 and (W-P)%S==0 "
            f"(got H={H}, W={W}, P={P}, S={S})"
        )

        feat = self.proj(x_planes)  # (B, D, h, w)
        h, w = feat.shape[2], feat.shape[3]
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, D)
        return tokens, (h, w)

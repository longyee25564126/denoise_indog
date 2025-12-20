import torch
from torch import nn


class ConvPatchTokenizer(nn.Module):
    """
    Simple conv-based patch tokenizer: (B, C, H, W) -> (B, N, D) tokens.
    """

    def __init__(self, in_ch: int, embed_dim: int = 256, patch_size: int = 8, bias: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x_planes: torch.Tensor):
        B, C, H, W = x_planes.shape
        P = self.patch_size
        assert C == self.in_ch, f"Expected {self.in_ch} channels, got {C}"
        assert H % P == 0 and W % P == 0, "Input spatial size must be divisible by patch_size"

        feat = self.proj(x_planes)  # (B, D, h, w)
        h, w = feat.shape[2], feat.shape[3]
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, D)
        return tokens, (h, w)

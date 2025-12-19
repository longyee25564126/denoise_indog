import torch
from torch import nn


class LSBMaskHead(nn.Module):
    """
    Predicts patch-wise mask from LSB tokens.
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else embed_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, lsb_tokens: torch.Tensor, grid_shape: tuple[int, int]) -> dict:
        B, N, D = lsb_tokens.shape
        h, w = grid_shape
        assert N == h * w, f"Token count {N} does not match grid {h}x{w}"

        m_logits_tok = self.mlp(lsb_tokens)  # (B, N, 1)
        m_logits = m_logits_tok.view(B, h, w, 1).permute(0, 3, 1, 2)  # (B,1,h,w)
        m_hat = torch.sigmoid(m_logits)
        m_hat_tok = m_hat.permute(0, 2, 3, 1).contiguous().view(B, N, 1)  # (B,N,1)

        return {"m_logits": m_logits, "m_hat": m_hat, "m_hat_tok": m_hat_tok}

# -*- coding: utf-8 -*-
"""
GatedMLP: A pure gated MLP baseline for REIMS spectral data.

The 1D Transformer applied to REIMS data uses seq_len=1 (the entire spectrum
is a single token). With a single token, self-attention degenerates:

    softmax(Q K^T / sqrt(d)) with shape [B, H, 1, 1] = 1.0 always

The attention output is therefore just V = x W_v, a plain linear transform.
The transformer is then functionally equivalent to:
    [Linear → FFN] × num_layers → pool → fc_out

GatedMLP makes this explicit: it applies modern building blocks
(RMSNorm, SwiGLU, residual connections, dropout) directly in input_dim space
with no large embedding projection. This isolates the contribution of the
architecture from any large fixed embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from .ttt_mixin import TTTMixin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class GatedMLPBlock(nn.Module):
    """RMSNorm → SwiGLU → Dropout → Residual, operating in input_dim space."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x)
        return residual + x


class GatedMLP(nn.Module, TTTMixin):
    """
    Gated MLP baseline.

    Operates directly in input_dim space, matching the effective computation
    of the Transformer on single-token REIMS spectra (where attention is
    degenerate and the FFN does all the real work).

    Set embed_dim > 0 to add a projection to a fixed embedding space first.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        embed_dim: int = 512,
        **kwargs,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim, bias=False)
        dim = embed_dim
        self.blocks = nn.ModuleList(
            [GatedMLPBlock(dim, dim * 2, dropout) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(dim)
        self.fc_out = nn.Linear(dim, output_dim, bias=False)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if x.dim() == 3:
            x = x.squeeze(1)

        x = self.proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.fc_out(x)

        if return_attention:
            return logits, []
        return logits


__all__ = ["GatedMLP"]

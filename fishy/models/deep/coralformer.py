# -*- coding: utf-8 -*-
"""
CoralFormer: A modern Gated MLP architecture for mass spectrometry.

Since mass spectra are highly sparse and permutation-variant (each bin is independent),
traditional Convolutions and sequence-based Attention often fail or scale poorly.
CoralFormer treats the entire spectrum as a holistic fingerprint, applying
modern deep learning advancements (RMSNorm, SwiGLU gating, residual connections)
to achieve superior performance with a fraction of the parameters of naive Transformers.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Union
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class CoralBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = RMSNorm(dim)
        # SwiGLU: w1 and w3 project to hidden_dim, w2 projects back
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        # SwiGLU Activation
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x)
        return residual + x


class CoralFormer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        # Project raw spectrum to a wide embedding space.
        # 512 provides massive expressive power; dropout=0.3 counteracts overfitting.
        embed_dim = 512
        self.proj = nn.Linear(input_dim, embed_dim, bias=False)

        self.blocks = nn.ModuleList(
            [CoralBlock(embed_dim, embed_dim * 2, dropout) for _ in range(num_layers)]
        )

        self.norm = RMSNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, output_dim, bias=False)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # Handle both [B, F] and [B, 1, F]
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


__all__ = ["CoralFormer"]

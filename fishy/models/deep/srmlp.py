# -*- coding: utf-8 -*-
"""
SRMLP: Sign-Random Gated MLP.
Features the SRNorm (Sign-Random Norm) layer which applies a fixed random
projection and sign-bit quantization to the spectral features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union


class SRNorm(nn.Module):
    """
    Sign-Random Normalization.
    Projects input to a random subspace and applies the sign function.
    This acts as a powerful non-linear regularizer and dimensionality compressor.
    """
    def __init__(self, dim: int, projection_dim: int = None):
        super().__init__()
        self.dim = dim
        self.projection_dim = projection_dim or dim
        
        # Fixed random projection (not learnable)
        self.register_buffer(
            "projection", 
            torch.randn(dim, self.projection_dim) * (self.projection_dim ** -0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, ..., dim]
        # 1. Project to random subspace
        x = x @ self.projection
        # 2. Sign operation (Quantization to -1, 0, 1)
        # We use a differentiable approximation or the straight-through estimator 
        # logic isn't needed here if we treat this as a static "encoding" step.
        return torch.sign(x)


class GatedMLPBlock(nn.Module):
    """Standard SwiGLU Gating Block."""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, dim * 2, bias=False)
        self.w2 = nn.Linear(dim * 2, dim, bias=False)
        self.w3 = nn.Linear(dim, dim * 2, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        # SwiGLU gating
        x = self.w2(F.silu(self.w1(x)) * self.w3(residual))
        return residual + self.drop(x)


class SRMLP(nn.Module):
    """
    Sign-Random Gated MLP.
    Combines SRNorm with high-capacity Gated MLP blocks.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Initial SRNorm to compress and robustify input
        self.sr_norm = SRNorm(input_dim, hidden_dim)
        
        # Gated processing backbone
        self.blocks = nn.ModuleList([
            GatedMLPBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        if x.dim() == 3:
            x = x.squeeze(1)
            
        # 1. Sign-Random Encoding
        x = self.sr_norm(x)
        
        # 2. Gated Deep Processing
        for block in self.blocks:
            x = block(x)
            
        x = self.final_norm(x)
        logits = self.fc_out(x)

        if return_attention:
            return logits, []
        return logits


__all__ = ["SRMLP", "SRNorm"]

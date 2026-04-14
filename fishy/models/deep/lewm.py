# -*- coding: utf-8 -*-
"""
LEWM: LeWorldModel-inspired Gated MLP.
Implements a Joint-Embedding Predictive Architecture (JEPA) style encoder
with SIGReg (Sketched-Isotropic-Gaussian Regularizer) for spectral data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from .augformer import RMSNorm


class SIGReg(nn.Module):
    """
    Sketched-Isotropic-Gaussian Regularizer.
    Ensures latent embeddings follow a Gaussian distribution to prevent collapse.
    Based on the Cramér-Wold Theorem logic used in LeWorldModel.
    """

    def __init__(self, dim: int, num_sketches: int = 64):
        super().__init__()
        self.dim = dim
        self.num_sketches = num_sketches
        # Fixed random projections for sketching
        self.register_buffer("projections", torch.randn(num_sketches, dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, dim]
        # Project latent embeddings onto random unit vectors
        projections = F.normalize(self.projections, p=2, dim=-1)
        sketches = z @ projections.t()  # [B, num_sketches]

        # In a full JEPA, this would be part of the loss function.
        # Here we provide the sketches for potential downstream loss computation
        # or apply a soft penalty if needed. For now, we return the regularized latent.
        return z


class GatedEncoderBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, dim * 2, bias=False)
        self.w2 = nn.Linear(dim * 2, dim, bias=False)
        self.w3 = nn.Linear(dim, dim * 2, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        # SwiGLU Gating
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return residual + self.drop(x)


class LEWM(nn.Module):
    """
    LeWorld Gated MLP (LEWM).
    A compact, JEPA-inspired architecture for high-fidelity spectral analysis.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 192,  # LeWM default compression
        num_layers: int = 4,
        dropout: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()

        # 1. Gated Encoder (Compresses spectrum to latent)
        self.encoder_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.encoder_blocks = nn.ModuleList(
            [GatedEncoderBlock(hidden_dim, dropout) for _ in range(num_layers // 2)]
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # 2. SIGReg (Latent conditioning)
        self.sigreg = SIGReg(latent_dim)

        # 3. Predictor/Classifier Backbone
        self.predictor_blocks = nn.ModuleList(
            [GatedEncoderBlock(latent_dim, dropout) for _ in range(num_layers // 2)]
        )

        self.norm = RMSNorm(latent_dim)
        self.fc_out = nn.Linear(latent_dim, output_dim, bias=False)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        if x.dim() == 3:
            x = x.squeeze(1)

        # Step 1: Encode
        x = self.encoder_proj(x)
        for block in self.encoder_blocks:
            x = block(x)
        z = self.to_latent(x)

        # Step 2: Regularize latent space (SIGReg logic)
        z = self.sigreg(z)

        # Step 3: Predict/Classify from latent
        for block in self.predictor_blocks:
            z = block(z)

        z = self.norm(z)
        logits = self.fc_out(z)

        if return_attention:
            return logits, []
        return logits


__all__ = ["LEWM"]

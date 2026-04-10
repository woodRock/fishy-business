# -*- coding: utf-8 -*-
"""
Role-Filler Network (RFN) for spectral analysis.

Enhanced version with Residual Connections, LayerNorm, and Dropout
for improved stability and regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


class RoleFillerBinding(nn.Module):
    """
    Enhanced Binding layer with Residual Connections and LayerNorm.
    Inspired by Vector Symbolic Architectures (VSA) and ResNets.
    """

    def __init__(self, dim: int, binding_type: str = "complex", dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.binding_type = binding_type
        self.filler_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, role: torch.Tensor, filler: torch.Tensor) -> torch.Tensor:
        # role: [B, N, D] or [1, N, D]
        # filler: [B, N, D] (Encoded intensities)

        # Project filler into binding space with activation
        f_proj = F.gelu(self.filler_proj(filler))

        if self.binding_type == "multiplicative":
            bound = role * f_proj
        elif self.binding_type == "additive":
            bound = role + f_proj
        elif self.binding_type == "complex":
            # Gated binding: soft interpolation preserves role signal even at low intensity
            gate = torch.sigmoid(role + f_proj)
            bound = gate * f_proj + (1 - gate) * role
        else:
            bound = role + f_proj

        # Residual Connection: Preserve original filler signal + bound relational context
        return self.norm(filler + self.dropout(bound))


class RoleFillerNet(nn.Module):
    """
    Fluid Relational Network (FRN) using Regularized Role-Filler relationships.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        binding_type: str = "complex",
        use_performer: bool = False,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        # Role Library: Learned vector for each input feature (m/z bin)
        self.role_embeddings = nn.Parameter(
            torch.randn(1, input_dim, hidden_dim) * 0.02
        )

        # Filler Encoder: Deep MLP to transform scalar intensity to vector space
        self.filler_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Two binding passes: initial association then a refinement step.
        # Fixed at 2 regardless of num_layers — binding is pre-processing,
        # not part of the relational engine depth.
        self.bindings = nn.ModuleList(
            [
                RoleFillerBinding(
                    hidden_dim, binding_type=binding_type, dropout=dropout
                )
                for _ in range(2)
            ]
        )

        # Relational Reasoning Engine (Stacked Encoders)
        if use_performer:
            from fishy.models.deep.performer import PerformerLayer

            self.relational_engine = nn.ModuleList(
                [
                    PerformerLayer(hidden_dim, heads=num_heads, dropout=dropout)
                    for _ in range(num_layers)
                ]
            )
            self.engine_type = "performer"
        else:
            # Multi-head Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-norm for better gradient flow
            )
            self.relational_engine = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            self.engine_type = "transformer"

        # CLS token for learned global aggregation (replaces mean pooling)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N] or [B, 1, N] or [B, N, 1]
        if x.dim() == 3:
            if x.shape[1] == 1:
                x = x.squeeze(1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)

        B, N = x.shape

        # 0. Log-transform intensities to compress dynamic range
        x = torch.log1p(x)

        # 1. Encode Fillers (intensities)
        x_seq = x.unsqueeze(-1)
        fillers = self.filler_encoder(x_seq)  # [B, N, D]

        # 2. Stacked Role-Filler Binding (progressive refinement)
        roles = self.role_embeddings.expand(B, -1, -1)  # [B, N, D]
        z = fillers
        for bind in self.bindings:
            z = bind(roles, z)

        # 2.5 Optional: Top-K Sparsity (Speed up + Denoising)
        if self.top_k is not None and self.top_k < N:
            # Select top-k on original (pre-log) intensities for peak-based filtering
            _, top_indices = torch.topk(x, self.top_k, dim=1)
            indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            z = torch.gather(z, 1, indices_expanded)

        # 3. Prepend CLS token for learned global aggregation
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        z = torch.cat([cls, z], dim=1)  # [B, N+1, D]

        # 4. Relational Reasoning (Induced Logic)
        if self.engine_type == "performer":
            for layer in self.relational_engine:
                z = layer(z)
        else:
            z = self.relational_engine(z)

        # 5. Extract CLS token output as global representation
        z = z[:, 0]  # [B, D]
        z = self.ln_out(z)
        z = self.dropout(z)

        # 6. Output Prediction
        return self.head(z)


__all__ = ["RoleFillerNet", "RoleFillerBinding"]

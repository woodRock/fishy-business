# -*- coding: utf-8 -*-
"""
Role-Filler Network (RFN) for spectral analysis.

This model is inspired by concepts of fluid intelligence, specifically the
capacity to store and manipulate role-filler relationships. It explicitly
binds feature-specific "roles" (learned embeddings for each m/z bin) with
their "fillers" (encoded intensity values) and processes them via a
relational attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


class RoleFillerBinding(nn.Module):
    """
    Binding layer that combines a Role (position/meaning) and a Filler (value).
    Inspired by Vector Symbolic Architectures (VSA).
    """

    def __init__(self, dim: int, binding_type: str = "multiplicative"):
        super().__init__()
        self.dim = dim
        self.binding_type = binding_type
        self.filler_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, role: torch.Tensor, filler: torch.Tensor) -> torch.Tensor:
        # role: [B, N, D] or [1, N, D]
        # filler: [B, N, D]
        
        filler = self.filler_proj(filler)
        
        if self.binding_type == "multiplicative":
            bound = role * filler
        elif self.binding_type == "additive":
            bound = role + filler
        elif self.binding_type == "complex":
            # Gated binding
            gate = torch.sigmoid(role + filler)
            bound = gate * filler + (1 - gate) * role
        else:
            bound = role + filler
            
        return self.norm(bound)


class RoleFillerNet(nn.Module):
    """
    Fluid Relational Network (FRN) using Role-Filler relationships.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        binding_type: str = "multiplicative",
        use_performer: bool = False,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        # Role Embeddings: Learned vector for each input feature (m/z bin)
        self.role_embeddings = nn.Parameter(torch.randn(1, input_dim, hidden_dim))
        
        # Filler Encoder: Transforms scalar intensity to vector space
        self.filler_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Binding mechanism
        self.binding = RoleFillerBinding(hidden_dim, binding_type=binding_type)

        # Relational Reasoning Engine
        if use_performer:
            # For very large input_dim, use Performer (linear attention)
            from fishy.models.deep.performer import PerformerLayer
            self.relational_engine = nn.ModuleList([
                PerformerLayer(hidden_dim, heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.engine_type = "performer"
        else:
            # Standard Transformer for relational reasoning
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True
            )
            self.relational_engine = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.engine_type = "transformer"

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N] or [B, 1, N] or [B, N, 1]
        if x.dim() == 3:
            if x.shape[1] == 1:
                x = x.squeeze(1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
        
        B, N = x.shape
        
        # 1. Encode Fillers (intensities)
        # [B, N] -> [B, N, 1]
        x_seq = x.unsqueeze(-1)
        fillers = self.filler_encoder(x_seq) # [B, N, D]
        
        # 2. Bind Roles and Fillers
        # roles: [1, N, D]
        roles = self.role_embeddings
        z = self.binding(roles, fillers) # [B, N, D]
        
        # 2.5 Optional: Top-K Sparsity (Speed up)
        # Only process the most intense peaks to avoid quadratic bottleneck
        if self.top_k is not None and self.top_k < N:
            # Get indices of top-k intensity values for each sample in batch
            _, top_indices = torch.topk(x, self.top_k, dim=1) # [B, K]
            
            # Use gather to select only those tokens for the relational engine
            # Expand indices for the hidden dimension
            indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            z = torch.gather(z, 1, indices_expanded) # [B, K, D]
        
        # 3. Relational Reasoning
        if self.engine_type == "performer":
            for layer in self.relational_engine:
                z = layer(z)
        else:
            z = self.relational_engine(z)
            
        # 4. Global Aggregation
        # Pool across feature tokens
        z = z.mean(dim=1)
        z = self.ln_out(z)
        
        # 5. Output Prediction
        return self.head(z)


__all__ = ["RoleFillerNet", "RoleFillerBinding"]

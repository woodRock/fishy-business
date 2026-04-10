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

    def __init__(self, dim: int, binding_type: str = "multiplicative", dropout: float = 0.1):
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
            # Gated binding
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
        binding_type: str = "multiplicative",
        use_performer: bool = False,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        # Role Library: Learned vector for each input feature (m/z bin)
        self.role_embeddings = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)
        
        # Filler Encoder: Deep MLP to transform scalar intensity to vector space
        self.filler_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Binding mechanism with internal regularization
        self.binding = RoleFillerBinding(hidden_dim, binding_type=binding_type, dropout=dropout)

        # Relational Reasoning Engine (Stacked Encoders)
        if use_performer:
            from fishy.models.deep.performer import PerformerLayer
            self.relational_engine = nn.ModuleList([
                PerformerLayer(hidden_dim, heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])
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
                norm_first=True # Pre-norm for better gradient flow
            )
            self.relational_engine = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.engine_type = "transformer"

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
        
        # 1. Encode Fillers (intensities)
        x_seq = x.unsqueeze(-1)
        fillers = self.filler_encoder(x_seq) # [B, N, D]
        
        # 2. Bind Roles and Fillers
        roles = self.role_embeddings
        z = self.binding(roles, fillers) # [B, N, D]
        
        # 2.5 Optional: Top-K Sparsity (Speed up + Denoising)
        if self.top_k is not None and self.top_k < N:
            _, top_indices = torch.topk(x, self.top_k, dim=1)
            indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            z = torch.gather(z, 1, indices_expanded)
        
        # 3. Relational Reasoning (Induced Logic)
        if self.engine_type == "performer":
            for layer in self.relational_engine:
                z = layer(z)
        else:
            z = self.relational_engine(z)
            
        # 4. Global Aggregation (Pooling)
        z = z.mean(dim=1)
        z = self.ln_out(z)
        z = self.dropout(z)
        
        # 5. Output Prediction
        return self.head(z)


__all__ = ["RoleFillerNet", "RoleFillerBinding"]

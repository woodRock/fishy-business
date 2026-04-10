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

        # Role Library: Learned vector for each input feature (m/z bin).
        # Initialised with Xavier-scale std so roles contribute meaningfully
        # from the first forward pass (multiplicative binding: bound = role * filler).
        self.role_embeddings = nn.Parameter(
            torch.randn(1, input_dim, hidden_dim) * (2.0 / hidden_dim) ** 0.5
        )

        # Filler Encoder: maps scalar log-intensity to hidden space.
        # No LayerNorm here: LN normalises across the hidden dim *per token*,
        # which maps every m/z bin to the same direction (since all bins share
        # the same Linear weights and log1p is always >= 0), destroying all
        # intensity information before binding.
        self.filler_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Single binding pass. With large N (e.g. 2080 m/z bins) and sum
        # pooling, gradient already flows fully to each token; stacking
        # binding passes only lengthens the path without adding capacity.
        self.binding = RoleFillerBinding(hidden_dim, binding_type=binding_type, dropout=dropout)

        # Relational Reasoning Engine
        # Three modes depending on configuration:
        #
        # "mlp"         (default) — position-wise MLP applied per token before
        #               mean-pooling. Processes all N tokens in O(N × D²) with
        #               constant memory. No cross-token attention, no OOM.
        #
        # "transformer" — standard quadratic attention. Requires top_k to be set
        #               to keep sequence length small enough to fit in memory.
        #
        # "performer"   — linear FAVOR+ attention. Still O(N × D) memory but
        #               MPS intermediates become large for N > ~1000; use only
        #               with top_k or on CUDA.
        if use_performer:
            from fishy.models.deep.performer import PerformerLayer

            self.relational_engine = nn.ModuleList(
                [
                    PerformerLayer(hidden_dim, heads=num_heads, dropout=dropout)
                    for _ in range(num_layers)
                ]
            )
            self.engine_type = "performer"
        elif top_k is not None:
            # Transformer over a short top_k sequence
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.relational_engine = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            self.engine_type = "transformer"
        else:
            # Position-wise MLP: O(N × D²) memory, works on full spectrum
            self.relational_engine = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.engine_type = "mlp"

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

        # 1. Peak selection FIRST — select top-k positions before encoding or binding.
        #    This mirrors peak picking in analytical chemistry and ensures:
        #    - Only informative peaks flow through the (expensive) encoder and binding
        #    - Role embeddings for selected positions get dense gradient updates
        #    - Sequence length entering the transformer is bounded to top_k + 1
        if self.top_k is not None and self.top_k < N:
            _, top_indices = torch.topk(x, self.top_k, dim=1)           # [B, K]
            x = torch.gather(x, 1, top_indices)                          # [B, K]
            idx_exp = top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            roles = torch.gather(
                self.role_embeddings.expand(B, -1, -1), 1, idx_exp
            )                                                             # [B, K, D]
        else:
            roles = self.role_embeddings.expand(B, -1, -1)               # [B, N, D]

        # 2. Encode Fillers (intensities → hidden space)
        fillers = self.filler_encoder(x.unsqueeze(-1))                   # [B, K/N, D]

        # 3. Role-filler binding
        z = self.binding(roles, fillers)

        # 4. Relational Reasoning + Global Aggregation
        # Sum pooling (not mean) so each of the N tokens receives the full
        # output gradient rather than a 1/N fraction of it. LayerNorm below
        # normalises the resulting scale.
        if self.engine_type == "performer":
            for layer in self.relational_engine:
                z = layer(z)
            z = z.sum(dim=1)
        elif self.engine_type == "transformer":
            z = self.relational_engine(z)
            z = z.sum(dim=1)
        else:
            # MLP: position-wise refinement, then sum-pool over all N tokens
            z = self.relational_engine(z)   # [B, N, D]
            z = z.sum(dim=1)                # [B, D]
        z = self.ln_out(z)
        z = self.dropout(z)

        # 6. Output Prediction
        return self.head(z)


__all__ = ["RoleFillerNet", "RoleFillerBinding"]

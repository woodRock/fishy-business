# -*- coding: utf-8 -*-
"""
Predictive Binding Network (PBN) v14 - Re-entrant Loop Engine.

Restored Version (Peak Performance):
1. Re-entrant Connections: Feedback loops from higher layers to lower layers.
2. Stochastic Recurrence: Probabilistic looping to prevent infinite recursion.
3. Backpropagation Through Time: Learning to refine perception over multiple 'Passes'.
4. Multiplicative Feedback: Global Gist filters sensory bindings for the next pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Tuple, List


class PrecisionWeightedBlock(nn.Module):
    """Refines context with gated relational updates."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.25):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, local_bindings: torch.Tensor, gist: torch.Tensor) -> torch.Tensor:
        x, _ = self.attn(self.norm(gist), self.norm(local_bindings), self.norm(local_bindings))
        return gist + self.ffn(self.norm(x)) * self.gate(gist)


class PBN(nn.Module):
    """
    Predictive Binding Network (v14).
    The 'Recurrent' version featuring Top-Down Feedback loops.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        top_k: int = 448,
        max_loops: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.max_loops = max_loops
        
        # 1. ENCODERS
        self.turbo_basis = nn.Parameter(torch.randn(input_dim, hidden_dim))
        nn.init.orthogonal_(self.turbo_basis)
        self.population_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.Softmax(dim=-1)
        )
        
        # Learnable harmonics
        self.freq_scales = nn.Parameter(torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)))
        
        # 2. CORE ENGINE
        self.core_block = PrecisionWeightedBlock(hidden_dim, num_heads, dropout)
        
        # Feedback Projector: Transforms high-level state back into input space
        self.feedback_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(2)

        x = torch.log1p(x.clamp(min=0.0))
        B, C = x.shape

        # --- INITIAL SENSORY ENCODING ---
        gist = torch.tanh(x @ self.turbo_basis).unsqueeze(1)
        
        top_val, top_idx = torch.topk(x, self.top_k, dim=1)
        fillers = self.population_encoder(top_val.unsqueeze(-1))
        times = top_idx.unsqueeze(-1).float() * self.freq_scales.view(1, 1, -1)
        roles = torch.cat([torch.sin(times), torch.cos(times)], dim=-1)
        local_bindings = fillers * roles

        # --- RE-ENTRANT LOOPING ---
        current_state = gist
        
        for i in range(self.max_loops):
            # Stochastic Exit during training
            if i > 0 and self.training and random.random() < (0.3 * i):
                break
            
            # Feed-forward Pass
            new_state = self.core_block(local_bindings, current_state)
            
            # Top-Down Feedback: The new state 'filters' the local bindings for the next pass
            feedback = self.feedback_projector(new_state)
            local_bindings = local_bindings * (1.0 + feedback)
            
            current_state = new_state

        # --- READOUT ---
        return self.head(current_state.squeeze(1))

    def binding_loss(self) -> torch.Tensor:
        w = self.turbo_basis
        res = torch.matmul(w.t(), w) - torch.eye(self.hidden_dim, device=w.device)
        return torch.mean(res**2) * 0.05

__all__ = ["PBN"]

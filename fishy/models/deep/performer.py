# -*- coding: utf-8 -*-
"""
Performer model for spectral classification.

Performers use a linear approximation of the attention mechanism via
Random Fourier Features (FAVOR+), allowing them to scale to very long sequences.

References:
1. Choromanski, K., et al. (2020). Rethinking Attention with Performers. arXiv:2009.14794.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_kernel(data, projection_matrix, is_query, epsilon=1e-6):
    """
    Softmax kernel approximation for Performer.
    """
    if projection_matrix is None:
        return data

    # Projection
    data_normalizer = (data.shape[-1]) ** -0.25
    data = data * data_normalizer

    # b: batch, h: heads, l: seq_len, d: head_dim, m: nb_features
    projection = torch.einsum("bhld,md->bhlm", data, projection_matrix)

    # Kernel computation
    data_squared_norm = torch.sum(data**2, dim=-1, keepdim=True) / 2

    return torch.exp(projection - data_squared_norm + epsilon)


class FastAttention(nn.Module):
    """
    FAVOR+ Fast Attention implementation.
    """

    def __init__(self, dim, heads=8, nb_features=64):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.nb_features = nb_features
        self.head_dim = dim // heads

        # Random projection matrix
        self.register_buffer(
            "projection_matrix", torch.randn(nb_features, self.head_dim)
        )

    def forward(self, q, k, v):
        # q, k, v: (batch, heads, seq_len, head_dim)

        # Map to random feature space
        q_prime = softmax_kernel(q, self.projection_matrix, is_query=True)
        k_prime = softmax_kernel(k, self.projection_matrix, is_query=False)

        # KV: (B, H, M, D)
        kv = torch.einsum("bhlm,bhld->bhmd", k_prime, v)

        # Normalization factor
        k_sum = k_prime.sum(dim=2)  # (B, H, M)
        z_raw = torch.einsum("bhlm,bhm->bhl", q_prime, k_sum)
        z = 1 / (z_raw + 1e-6)

        # Result: (B, H, L, D)
        out = torch.einsum("bhlm,bhmd,bhl->bhld", q_prime, kv, z)

        return out


class PerformerLayer(nn.Module):
    def __init__(self, dim, heads=8, nb_features=64, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = FastAttention(dim, heads, nb_features)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.num_heads = heads

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        residual = x
        x = self.ln1(x)

        q = self.q_proj(x).view(B, L, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, -1).transpose(1, 2)

        x = self.attn(q, k, v)
        x = x.transpose(1, 2).reshape(B, L, D)
        x = residual + self.out_proj(x)

        x = x + self.mlp(self.ln2(x))
        return x


class Performer(nn.Module):
    """
    Performer model for spectral classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_heads: int = 8,
        nb_features: int = 64,
        **kwargs,
    ) -> None:
        super(Performer, self).__init__()

        self.embedding = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList(
            [
                PerformerLayer(
                    hidden_dim,
                    heads=num_heads,
                    nb_features=nb_features,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)

        x = self.ln_out(x)
        x = x.mean(dim=1)
        return self.head(x)


__all__ = ["Performer", "PerformerLayer"]

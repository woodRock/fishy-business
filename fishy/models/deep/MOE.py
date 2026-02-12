# -*- coding: utf-8 -*-
"""
Mixture of Experts (MOE) Transformer model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_queries = nn.Linear(d_model, d_model)
        self.w_keys = nn.Linear(d_model, d_model)
        self.w_values = nn.Linear(d_model, d_model)
        self.w_output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        queries = self.w_queries(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        keys = self.w_keys(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        values = self.w_values(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, values).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_output(out)


class Expert(nn.Module):
    """
    Expert network within the MOE layer.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MOELayer(nn.Module):
    """
    Mixture of Experts layer.
    """

    def __init__(self, d_model: int, num_experts: int, d_ff: int, dropout: float = 0.1):
        super(MOELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        gate_scores = F.softmax(self.gate(x_flat), dim=-1)
        
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        out = torch.bmm(gate_scores.unsqueeze(1), expert_outputs).squeeze(1)
        return out.view(batch_size, seq_len, d_model)


class MOE(nn.Module):
    """
    MOE model for spectral data classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        num_heads: int = 4,
        num_experts: int = 4,
        **kwargs
    ) -> None:
        """
        Initializes the MOE model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden dimension. Defaults to 128.
            num_layers (int, optional): Number of layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            num_experts (int, optional): Number of experts. Defaults to 4.
        """
        super(MOE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "attn": MultiHeadAttention(hidden_dim, num_heads, dropout),
                "moe": MOELayer(hidden_dim, num_experts, hidden_dim * 4, dropout),
                "norm1": nn.LayerNorm(hidden_dim),
                "norm2": nn.LayerNorm(hidden_dim)
            }))

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        x = self.embedding(x)
        for layer in self.layers:
            x = layer["norm1"](x + layer["attn"](x))
            x = layer["norm2"](x + layer["moe"](x))
        
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc_out(x)

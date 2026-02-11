"""Mixture of Experts (MoE) Transformer model.

This module implements a Transformer architecture with a Mixture of Experts (MoE) layer
replacing the standard feed-forward network. The MoE layer allows for dynamic routing of
inputs to multiple expert networks, enabling the model to learn complex representations
while maintaining computational efficiency.


References:

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
    A. N., ... & Polosukhin, I. (2017).
    Attention is all you need.
    Advances in neural information processing systems, 30.
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for the Transformer model.

    Attributes:
        input_dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        qkv (nn.Linear): Combined projection for Q, K, and V.
        fc_out (nn.Linear): Final output projection.
        scale (float): Scaling factor for dot-product attention.
    """

    def __init__(self, input_dim: int, num_heads: int) -> None:
        """
        Initializes the MultiHeadAttention layer.

        Args:
            input_dim (int): Number of input features.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        assert input_dim % num_heads == 0

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Combined projection for Q, K, V
        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)

        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, input_dim).
        """
        batch_size = x.shape[0]

        # Single matrix multiplication for all projections
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.input_dim)
        x = self.fc_out(x)
        return x


class ExpertLayer(nn.Module):
    """
    Individual expert neural network.

    Args:
        input_dim (int): Input dimensionality.
        hidden_dim (int): Hidden layer dimensionality.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Expert output.
        """
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) layer for routing inputs to multiple experts.

    Attributes:
        experts (nn.ModuleList): List of expert layers.
        gate (nn.Linear): Gating network for routing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        k: int = 2,
        dropout: float = 0.1,
        use_majority_voting: bool = False,
    ) -> None:
        """
        Initializes the MoE layer.

        Args:
            input_dim (int): Input feature size.
            hidden_dim (int): Intermediate dimension for experts.
            num_experts (int, optional): Total number of experts. Defaults to 4.
            k (int, optional): Number of experts to activate per input. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            use_majority_voting (bool, optional): If True, uses all experts and averages. Defaults to False.
        """
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.use_majority_voting = use_majority_voting

        # Create experts
        self.experts = nn.ModuleList(
            [ExpertLayer(input_dim, hidden_dim, dropout) for _ in range(num_experts)]
        )

        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)

        # Expert usage tracking
        self.expert_usage_counts = defaultdict(int)
        self.total_tokens = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input sequence tensor.

        Returns:
            torch.Tensor: Combined output from experts.
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        if self.use_majority_voting:
            expert_outputs = []
            for expert in self.experts:
                expert_out = expert(x_flat)
                expert_outputs.append(expert_out)

            self.total_tokens += x_flat.size(0)
            for i in range(self.num_experts):
                self.expert_usage_counts[i] += x_flat.size(0)

            combined_output = torch.stack(expert_outputs).mean(dim=0)

        else:
            gates = self.gate(x_flat)
            gate_scores, expert_indices = torch.topk(gates, self.k, dim=-1)
            gate_scores = F.softmax(gate_scores, dim=-1)

            for i in range(self.num_experts):
                self.expert_usage_counts[i] += torch.sum(expert_indices == i).item()
            self.total_tokens += expert_indices.numel()

            final_output = torch.zeros_like(x_flat)
            flat_expert_indices = expert_indices.flatten()
            flat_gate_scores = gate_scores.flatten()

            batch_indices = torch.arange(
                x_flat.size(0), device=x_flat.device
            ).repeat_interleave(self.k)

            for i, expert in enumerate(self.experts):
                expert_mask = flat_expert_indices == i
                if expert_mask.any():
                    selected_batch_indices = batch_indices[expert_mask]
                    selected_gate_scores = flat_gate_scores[expert_mask].unsqueeze(1)
                    expert_input = x_flat[selected_batch_indices]
                    expert_output = expert(expert_input)
                    final_output.index_add_(
                        0, selected_batch_indices, expert_output * selected_gate_scores
                    )

            combined_output = final_output

        return combined_output.view(batch_size, seq_len, d_model)

    def get_expert_utilization(self) -> List[float]:
        """
        Returns the utilization percentage for each expert.

        Returns:
            List[float]: Utilization ratios.
        """
        total = sum(self.expert_usage_counts.values())
        if total == 0:
            return [0.0] * self.num_experts
        return [self.expert_usage_counts[i] / total for i in range(self.num_experts)]


class MOE(nn.Module):
    """
    Transformer model with Mixture of Experts (MoE) instead of feed-forward layers.

    Attributes:
        embedding (nn.Linear): Initial feature projection.
        attention_layers (nn.ModuleList): Attention blocks.
        moe_layers (nn.ModuleList): MoE blocks.
        fc_out (nn.Linear): Classification head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int = 1,
        num_experts: int = 4,
        k: int = 2,
        dropout: float = 0.1,
        use_majority_voting: bool = False,
    ) -> None:
        """
        Initializes the MOE model.

        Args:
            input_dim (int): Input feature size.
            output_dim (int): Number of output classes.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Model hidden dimension.
            num_layers (int, optional): Number of blocks. Defaults to 1.
            num_experts (int, optional): Number of experts per block. Defaults to 4.
            k (int, optional): Top-k experts to route to. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            use_majority_voting (bool, optional): Enable majority voting mode. Defaults to False.
        """
        super().__init__()

        self.d_model = hidden_dim
        if self.d_model % num_heads != 0:
            self.d_model = ((self.d_model // num_heads) + 1) * num_heads

        self.embedding = nn.Linear(input_dim, self.d_model)

        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(self.d_model, num_heads) for _ in range(num_layers)]
        )

        self.moe_layers = nn.ModuleList(
            [
                MixtureOfExperts(
                    input_dim=self.d_model,
                    hidden_dim=hidden_dim,
                    num_experts=num_experts,
                    k=k,
                    dropout=dropout,
                    use_majority_voting=use_majority_voting,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(self.d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input features of shape (B, D).

        Returns:
            torch.Tensor: Output logits.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.embedding(x)

        for attention, moe in zip(self.attention_layers, self.moe_layers):
            residual = x
            x = self.layer_norm1(x)
            x = residual + self.dropout(attention(x))

            residual = x
            x = self.layer_norm2(x)
            x = residual + self.dropout(moe(x))

        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x


__all__ = ["MOE"]

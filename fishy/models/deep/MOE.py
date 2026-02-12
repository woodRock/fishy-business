# -*- coding: utf-8 -*-
"""
Mixture of Experts (MoE) model for spectral data.

This model uses a gating network to dynamically weight the outputs of multiple
expert networks (Transformers) based on the input spectrum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Transformer


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) using Transformer experts.

    Attributes:
        experts (nn.ModuleList): List of expert Transformer models.
        gate (nn.Linear): Gating network to determine weights for each expert.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_experts: int = 3,
        **kwargs
    ) -> None:
        """
        Initializes the MixtureOfExperts model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes/dimensions.
            hidden_dim (int, optional): Hidden dimension. Defaults to 128.
            num_layers (int, optional): Layers per expert. Defaults to 4.
            num_heads (int, optional): Heads per expert. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            num_experts (int, optional): Number of expert networks. Defaults to 3.
        """
        super().__init__()

        self.experts = nn.ModuleList([
            Transformer(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, 
                        num_layers=num_layers, num_heads=num_heads, dropout=dropout)
            for _ in range(num_experts)
        ])

        # Gating network to learn weighting of experts
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        """
        # Get gating weights
        gate_input = x.mean(dim=1) if x.dim() == 3 else x
        weights = F.softmax(self.gate(gate_input), dim=-1) # (batch_size, num_experts)

        # Get expert outputs
        expert_outputs = [expert(x) for expert in self.experts] # List of (batch_size, output_dim)
        expert_outputs = torch.stack(expert_outputs, dim=1) # (batch_size, num_experts, output_dim)

        # Weighted sum of expert outputs
        weighted_output = torch.bmm(weights.unsqueeze(1), expert_outputs).squeeze(1)

        return weighted_output


__all__ = ["MixtureOfExperts"]

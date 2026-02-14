# -*- coding: utf-8 -*-
"""
Sparsely-Gated Mixture of Experts (GMOE) for spectral data.

This model uses a top-k gating network to sparsely activate expert networks
(Transformers) based on the input spectrum, improving efficiency and specialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Transformer


class SparselyGatedMoE(nn.Module):
    """
    Sparsely-Gated Mixture of Experts (GMOE) using Transformer experts.

    Attributes:
        experts (nn.ModuleList): List of expert Transformer models.
        gate (nn.Linear): Gating network to determine weights for each expert.
        k (int): Number of top experts to activate per sample.
        output_dim (int): Number of output classes/dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_experts: int = 8,
        k: int = 2,
        **kwargs,
    ) -> None:
        """
        Initializes the SparselyGatedMoE model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes/dimensions.
            hidden_dim (int, optional): Hidden dimension. Defaults to 128.
            num_layers (int, optional): Layers per expert. Defaults to 4.
            num_heads (int, optional): Heads per expert. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            num_experts (int, optional): Total number of expert networks. Defaults to 8.
            k (int, optional): Number of experts to activate. Defaults to 2.
        """
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.output_dim = output_dim

        self.experts = nn.ModuleList(
            [
                Transformer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with sparse routing.
        """
        # Get gating scores
        # x shape: (batch_size, input_dim) or (batch_size, 1, input_dim)
        gate_input = x.mean(dim=1) if x.dim() == 3 else x
        logits = self.gate(gate_input)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        
        # Softmax over the top-k selected experts
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Initialize output
        batch_size = x.shape[0]
        
        # Expert results accumulator
        expert_results = torch.zeros(batch_size, self.output_dim, device=x.device)

        # We can optimize slightly by only running experts that were selected at least once
        unique_indices = torch.unique(top_k_indices)
        
        for expert_idx in unique_indices:
            # Find which samples in the batch selected this expert
            mask = (top_k_indices == expert_idx)
            sample_indices, k_pos = torch.where(mask)
            
            if len(sample_indices) > 0:
                # Run the expert on the relevant samples
                expert_out = self.experts[expert_idx](x[sample_indices])
                # Weight the output by the gate score at the corresponding k position
                weights = top_k_weights[sample_indices, k_pos].unsqueeze(1)
                expert_results[sample_indices] += expert_out * weights

        return expert_results


__all__ = ["SparselyGatedMoE"]

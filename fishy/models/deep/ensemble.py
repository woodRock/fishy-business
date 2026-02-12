# -*- coding: utf-8 -*-
"""
Multi-Scale/Resolution Ensemble of Transformers.

This model ensembles three Transformer architectures with varying depths and attention heads:
1. Small: 2 layers, 2 heads
2. Medium: 4 layers, 4 heads
3. Large: 8 layers, 8 heads

The outputs are concatenated and passed through a final classification/regression head.
"""

import torch
import torch.nn as nn
from .transformer import Transformer


class MultiScaleTransformerEnsemble(nn.Module):
    """
    An ensemble of Transformers with different scales/resolutions.

    Attributes:
        experts (nn.ModuleList): List of Transformer models with different configurations.
        classifier (nn.Sequential): Final classification/regression head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        """
        Initializes the MultiScaleTransformerEnsemble model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes/dimensions.
            hidden_dim (int, optional): Hidden dimension for transformers. Defaults to 128.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()

        # Define 3 members with different scales
        # We set output_dim to hidden_dim for each so we can concat them
        self.experts = nn.ModuleList([
            Transformer(input_dim=input_dim, output_dim=hidden_dim, hidden_dim=hidden_dim, 
                        num_layers=2, num_heads=2, dropout=dropout),
            Transformer(input_dim=input_dim, output_dim=hidden_dim, hidden_dim=hidden_dim, 
                        num_layers=4, num_heads=4, dropout=dropout),
            Transformer(input_dim=input_dim, output_dim=hidden_dim, hidden_dim=hidden_dim, 
                        num_layers=8, num_heads=8, dropout=dropout)
        ])

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input spectrum of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Ensemble output.
        """
        # Get expert features
        expert_features = [expert(x) for expert in self.experts] # List of (batch_size, hidden_dim)
        
        # Concatenate features
        combined = torch.cat(expert_features, dim=1) # (batch_size, hidden_dim * 3)

        # Final prediction
        return self.classifier(combined)


__all__ = ["MultiScaleTransformerEnsemble"]

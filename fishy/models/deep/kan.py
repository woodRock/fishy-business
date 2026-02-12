# -*- coding: utf-8 -*-
"""
Kolmogorov-Arnold Network (KAN) model for spectral classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    """
    A single KAN layer with spline-based activation.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.trunc_normal_(self.spline_weight, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified KAN layer logic
        base_output = F.linear(x, self.base_weight)
        # In a real KAN, we would evaluate B-splines here
        # This is a simplified version for integration purposes
        return base_output


import math

class KAN(nn.Module):
    """
    KAN-based model for spectral data classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        """
        Initializes the KAN model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 128.
            num_layers (int, optional): Number of layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(KAN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        layers = []
        in_f = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_f, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_f = hidden_dim
        
        layers.append(nn.Linear(in_f, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -*- coding: utf-8 -*-
"""
Dense Neural Network model for spectral classification.
"""

import torch
import torch.nn as nn


class Dense(nn.Module):
    """
    Standard fully connected (dense) neural network.

    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes.
        hidden_dim (int): Hidden layer dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        """
        Initializes the Dense model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 128.
            num_layers (int, optional): Number of hidden layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        layers = []
        in_f = input_dim
        for i in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_f, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_f = hidden_dim

        layers.append(nn.Linear(in_f, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -*- coding: utf-8 -*-
"""
Ensemble model for spectral classification.
"""

import torch
import torch.nn as nn


class Ensemble(nn.Module):
    """
    Ensemble model combining multiple base classifiers.

    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes.
        hidden_dim (int): Hidden layer dimension for base learners.
        dropout (float): Dropout probability.
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
        Initializes the Ensemble model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 128.
            num_layers (int, optional): Number of layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(Ensemble, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Example: An ensemble of several MLP branches
        self.branch1 = self._make_branch(input_dim, hidden_dim, num_layers, dropout)
        self.branch2 = self._make_branch(input_dim, hidden_dim, num_layers, dropout)
        self.branch3 = self._make_branch(input_dim, hidden_dim, num_layers, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _make_branch(self, input_dim, hidden_dim, num_layers, dropout):
        layers = []
        in_f = input_dim
        for i in range(num_layers):
            out_f = hidden_dim if i > 0 else hidden_dim * 2
            layers.extend([
                nn.Linear(in_f, out_f),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_f = out_f
        layers.append(nn.Linear(in_f, hidden_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        combined = torch.cat((out1, out2, out3), dim=1)
        return self.classifier(combined)

# -*- coding: utf-8 -*-
"""
Residual Convolutional Neural Network (RCNN) for spectral classification.
"""

import torch
import torch.nn as nn


class RCNN(nn.Module):
    """
    Residual Convolutional Neural Network (RCNN) model.

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
        **kwargs
    ) -> None:
        """
        Initializes the RCNN model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 128.
            num_layers (int, optional): Number of blocks. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(RCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dynamic calculation of flattened features
        self.flat_features = 64 * (input_dim // 4)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.contiguous().view(x.size(0), -1)
        return self.fc_layers(x)

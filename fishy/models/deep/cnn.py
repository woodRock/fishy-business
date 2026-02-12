# -*- coding: utf-8 -*-
"""
Simple Convolutional Neural Network (CNN) model for spectral classification.
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Standard CNN architecture for 1D spectral data.
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
        Initializes the CNN model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden dimension. Defaults to 128.
            num_layers (int, optional): Number of layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.flat_f = 64 * (input_dim // 4)
        self.fc = nn.Sequential(
            nn.Linear(self.flat_f, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

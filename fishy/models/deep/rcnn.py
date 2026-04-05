# -*- coding: utf-8 -*-
"""
Residual Convolutional Neural Network (RCNN) for spectral classification.

This model uses residual blocks with 1D convolutions to capture spectral patterns
while allowing for deeper architectures via skip connections.
"""

import torch
from fishy.models.utils import ensure_conv_input
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += residual
        out = self.relu(out)
        return out


class RCNN(nn.Module):
    """
    Improved Residual Convolutional Neural Network (RCNN) model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super(RCNN, self).__init__()

        # Deeper initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks - using a bottle-neck like structure or more layers
        layers = []
        for _ in range(num_layers):
            layers.append(ResidualBlock(64, dropout=dropout))
        self.res_layers = nn.Sequential(*layers)

        # Global pooling and output
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        # Concatenate Avg and Max pooling for richer representation
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_conv_input(x)

        x = self.initial_conv(x)
        x = self.res_layers(x)
        
        avg_x = self.avgpool(x)
        max_x = self.maxpool(x)
        x = torch.cat([avg_x, max_x], dim=1)
        
        x = torch.flatten(x, 1)
        return self.fc(x)


__all__ = ["RCNN"]

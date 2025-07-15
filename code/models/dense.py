""" 
DenseNet-like model for classification tasks.

This module implements a DenseNet-like architecture for classification tasks.
It includes dense blocks, transition layers, and fully connected layers.
The architecture is designed to handle 1D input data, such as time series or sequential data.
It uses batch normalization, ReLU activation, and dropout for regularization.

References:
1. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017).
   Densely connected convolutional networks.
   In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).
2. Srivastava, N., Hinton, G., Krizhevsky, A., & Sutskever, I. (2014).
   Dropout: A simple way to prevent neural networks from overfitting.
   Journal of Machine Learning Research, 15(1), 1929-1958.
3. Ioffe, S., & Szegedy, C. (2015).
   Batch normalization: Accelerating deep network training by reducing internal covariate shift.
   In International conference on machine learning (pp. 448-456).
4. Kingma, D. P., & Ba, J. (2014).
   Adam: A method for stochastic optimization.
   In International conference on learning representations (ICLR).  s
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm


class DenseBlock(nn.Module):
    """ Dense block for DenseNet-like architecture. """
    def __init__(self, in_channels, growth_rate, num_layers, dropout=0.5) -> None:
        """ Initialize the dense block.

        This block consists of multiple convolutional layers with batch normalization,
        ReLU activation, and dropout for regularization. Each layer takes the output
        from all previous layers as input, promoting feature reuse.
        
        Args:
            in_channels (int): Number of input channels.
            growth_rate (int): Growth rate of the block.
            num_layers (int): Number of layers in the block.
            dropout (float): Dropout rate for regularization.

        """
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm1d(in_channels + i * growth_rate),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1
                ),
                nn.Dropout(p=dropout),
            )
            self.layers.append(layer)

    def forward(self, x):
        """ Forward pass through the dense block.
        
        Args: 
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, sequence_length),
            where out_channels is the number of input channels plus growth_rate times num_layers.
        """
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class TransitionLayer(nn.Module):
    """ Transition layer for DenseNet-like architecture."""
    def __init__(self, in_channels, out_channels, dropout=0.5) -> None:
        """ Initialize the transition layer.
        
        This layer consists of batch normalization, ReLU activation, a 1x1 convolution,
        dropout for regularization, and average pooling to reduce the number of channels.

        Args: 
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float): Dropout rate for regularization.
        """
        super(TransitionLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Dropout(p=dropout),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        """ Forward pass through the transition layer.
        
        Args: 
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, new_sequence_length),
            where new_sequence_length is reduced by half due to average pooling.
        """
        return self.layers(x)


class Dense(nn.Module):
    """ DenseNet-like model for classification tasks."""
    def __init__(self, input_dim, output_dim, dropout=0.3) -> None:
        """ Initialize the DenseNet-like model.
        
        Args: 
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            dropout (float): Dropout rate for regularization.   
        """
        super(Dense, self).__init__()

        # Initial convolution
        self.first_conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)

        # DenseNet configuration
        growth_rate = 16
        num_layers_per_block = 4

        # First dense block
        self.dense1 = DenseBlock(32, growth_rate, num_layers_per_block, dropout)
        num_channels = 32 + growth_rate * num_layers_per_block

        # Transition layer
        self.trans1 = TransitionLayer(num_channels, num_channels // 2, dropout)
        num_channels = num_channels // 2

        # Second dense block
        self.dense2 = DenseBlock(
            num_channels, growth_rate, num_layers_per_block, dropout
        )
        num_channels = num_channels + growth_rate * num_layers_per_block

        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(4)

        # Calculate the flattened features size
        self.flat_features = num_channels * 4

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        """ Forward pass through the DenseNet-like model.
        
        Args: 
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim, sequence_length).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_dim),
            where output_dim is the number of classes.
        """
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.first_conv(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x


__all__ = ["Dense"]  # Export the Dense model for use in other modules

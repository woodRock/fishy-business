"""Temporal Convolutional Network (TCN) for time series classification.

This module implements a Temporal Convolutional Network (TCN) for time series classification tasks.
It includes temporal blocks with dilated convolutions, batch normalization, ReLU activation,
and dropout for regularization. The architecture is designed to handle 1D input data, such as time series or sequential data.

References:
1. Bai, S., Kolter, J. Z., & Koltun, V. (2018).
   An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.
   arXiv preprint arXiv:1803.01271.
2. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
   Dropout: a simple way to prevent neural networks from overfitting.
   The journal of machine learning research, 15(1), 1929-1958.
3. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. R. (2012).
   Improving neural networks by preventing co-adaptation of feature detectors.
   arXiv preprint arXiv:1207.0580.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm


class Chomp1d(nn.Module):
    """
    Removes the last elements of a time series.
    Used to ensure causal convolutions for time series prediction.
    """

    def __init__(self, chomp_size) -> None:
        """Initialize the Chomp1d module.

        Args:
            chomp_size (int): Number of elements to remove from the end of the sequence.
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """Forward pass through the Chomp1d module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor with the last chomp_size elements removed from the sequence length dimension.
        """
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    """
    A temporal block consisting of dilated causal convolutions,
    non-linearities, and residual connections.
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ) -> None:
        """Initialize the TemporalBlock.

        Args:
            n_inputs (int): Number of input channels.
            n_outputs (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            dilation (int): Dilation factor for the convolution.
            padding (int): Padding size for the convolution.
            dropout (float): Dropout rate for regularization. Defaults to 0.2.
        """
        super(TemporalBlock, self).__init__()

        # First dilated convolution layer
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution layer
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Stack all operations in a sequential module
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.bn1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.bn2,
            self.relu2,
            self.dropout2,
        )

        # Residual connection if input and output dimensions differ
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights using Kaiming initialization."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """Forward pass through the temporal block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_outputs, new_sequence_length),
            where new_sequence_length is determined by the kernel size, stride, and dilation.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network composed of temporal blocks
    with increasing dilation factors.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2) -> None:
        """Initialize the TemporalConvNet.

        Args:
            num_inputs (int): Number of input channels.
            num_channels (list): List of output channels for each temporal block.
            kernel_size (int): Size of the convolutional kernel. Defaults to 2.
            dropout (float): Dropout rate for regularization. Defaults to 0.2.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the TCN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_channels[-1], new_sequence_length),
            where new_sequence_length is determined by the kernel size, stride, and dilation.
        """
        return self.network(x)


class TCN(nn.Module):
    """
    Complete TCN model with temporal blocks and fully connected layers
    """

    def __init__(
        self, input_dim, output_dim, num_channels=None, kernel_size=3, dropout=0.3
    ) -> None:
        """Initialize the TCN model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            num_channels (list): List of output channels for each temporal block. Defaults to None.
            kernel_size (int): Size of the convolutional kernel. Defaults to 3.
            dropout (float): Dropout rate for regularization. Defaults to 0.3.
        """
        super(TCN, self).__init__()

        # Default channel configuration if none provided
        if num_channels is None:
            num_channels = [32, 64, 64]  # Reduced complexity default configuration

        # Input normalization
        self.input_norm = nn.BatchNorm1d(1)

        # Temporal Convolutional Network
        self.tcn = TemporalConvNet(
            num_inputs=1,  # Single channel input
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Adaptive pooling and flattening
        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
        self.flatten = nn.Flatten()

        # Fully connected layers with batch normalization
        self.fc_layers = nn.Sequential(
            nn.Linear(num_channels[-1] * 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for fully connected layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def apply_weight_norm(self):
        """Apply weight normalization to all convolutional layers"""

        def _apply_weight_norm(module):
            if isinstance(module, nn.Conv1d):
                weight_norm(module)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization from all convolutional layers"""

        def _remove_weight_norm(module):
            try:
                if isinstance(module, nn.Conv1d):
                    nn.utils.remove_weight_norm(module)
            except ValueError:
                pass

        self.apply(_remove_weight_norm)

    def forward(self, x):
        """
        Forward pass through the complete TCN model

        Args:
            x: Input tensor of shape [batch_size, sequence_length]

        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Add channel dimension [batch_size, 1, sequence_length]
        x = x.unsqueeze(1)

        # Normalize input
        x = self.input_norm(x)

        # Pass through TCN
        x = self.tcn(x)

        # Pool and flatten
        x = self.adaptive_pool(x)
        x = self.flatten(x)

        # Pass through fully connected layers
        x = self.fc_layers(x)

        return x

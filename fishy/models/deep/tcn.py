"""Temporal Convolutional Network (TCN) for spectral classification.

This module implements a Temporal Convolutional Network (TCN) for processing
1D spectral data. It uses dilated causal convolutions to capture long-range
patterns without the recurrent overhead.


References:

1. Bai, S., Kolter, J. Z., & Koltun, V. (2018).
   An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.
   arXiv preprint arXiv:1803.01271.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List, Optional, Tuple


class Chomp1d(nn.Module):
    """
    Removes the last elements of a sequence to ensure causality.
    """

    def __init__(self, chomp_size: int) -> None:
        """
        Initializes the Chomp1d module.

        Args:
            chomp_size (int): Number of elements to remove from the end.
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).

        Returns:
            torch.Tensor: Chomped tensor of shape (B, C, L - chomp_size).
        """
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    """
    A temporal block consisting of two layers of dilated causal convolutions.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        """
        Initializes the TemporalBlock.

        Args:
            n_inputs (int): Number of input channels.
            n_outputs (int): Number of output channels.
            kernel_size (int): Convolution kernel size.
            stride (int): Stride of the convolution.
            dilation (int): Dilation factor.
            padding (int): Padding size.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(TemporalBlock, self).__init__()

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

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self) -> None:
        """Initializes weights using normal distribution."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Residual output.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network composed of several temporal blocks.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Initializes the TCN.

        Args:
            num_inputs (int): Number of input channels.
            num_channels (List[int]): List of output channels for each level.
            kernel_size (int, optional): Kernel size. Defaults to 2.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Feature map from the last block.
        """
        return self.network(x)


class TCN(nn.Module):
    """
    Complete TCN model with adaptive pooling and classification head.

    Attributes:
        input_norm (nn.BatchNorm1d): Input normalization layer.
        tcn (TemporalConvNet): TCN backbone.
        adaptive_pool (nn.AdaptiveMaxPool1d): Pooling to fixed size.
        fc_layers (nn.Sequential): Classification head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        """
        Initializes the TCN model.

        Args:
            input_dim (int): Input feature size.
            output_dim (int): Number of output classes.
            num_channels (Optional[List[int]], optional): Channels per TCN block. Defaults to None.
            kernel_size (int, optional): Convolution kernel size. Defaults to 3.
            dropout (float, optional): Dropout probability. Defaults to 0.3.
        """
        super(TCN, self).__init__()

        if num_channels is None:
            num_channels = [32, 64, 64]

        self.input_norm = nn.BatchNorm1d(1)

        self.tcn = TemporalConvNet(
            num_inputs=1,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(
            nn.Linear(num_channels[-1] * 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes weights using Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def apply_weight_norm(self) -> None:
        """Applies weight normalization to all convolutional layers."""

        def _apply_weight_norm(module):
            if isinstance(module, nn.Conv1d):
                weight_norm(module)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self) -> None:
        """Removes weight normalization from all convolutional layers."""

        def _remove_weight_norm(module):
            try:
                if isinstance(module, nn.Conv1d):
                    nn.utils.remove_weight_norm(module)
            except ValueError:
                pass

        self.apply(_remove_weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_norm(x)
        x = self.tcn(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

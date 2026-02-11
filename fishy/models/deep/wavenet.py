"""WaveNet model for spectral classification.

This model uses causal convolutions and residual blocks to process sequential data.
It includes an initial causal convolutional layer, multiple residual blocks for feature extraction,
and a fully connected layer for classification.


References:

1. van den Oord, A., Dieleman, S., Zen, H.,
   Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016).
   WaveNet: A generative model for raw audio.
   arXiv preprint arXiv:1609.03499.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List, Tuple, Any


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with support for dilations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        **kwargs,
    ) -> None:
        """
        Initializes the CausalConv1d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size.
            dilation (int, optional): Dilation factor. Defaults to 1.
            **kwargs: Additional arguments for Conv1d.
        """
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Causally padded output tensor.
        """
        x = self.conv(x)
        if self.padding != 0:
            return x[:, :, : -self.padding]
        return x


class ResidualBlock(nn.Module):
    """
    WaveNet residual block with skip connections and gated activations.
    """

    def __init__(
        self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.2
    ) -> None:
        """
        Initializes the ResidualBlock.

        Args:
            channels (int): Number of input/output channels.
            kernel_size (int): Convolution kernel size.
            dilation (int): Dilation factor.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(ResidualBlock, self).__init__()

        self.dilated_conv = CausalConv1d(
            channels, 2 * channels, kernel_size, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (residual_output, skip_output)
        """
        original = x

        # Dilated convolution
        x = self.dilated_conv(x)

        # Gated activation unit
        filter_x, gate_x = torch.chunk(x, 2, dim=1)
        x = torch.tanh(filter_x) * torch.sigmoid(gate_x)

        # 1x1 convolution
        x = self.conv_1x1(x)
        x = self.dropout(x)

        # Residual and skip connections
        residual = original + x

        return residual, x


class WaveNet(nn.Module):
    """
    WaveNet architecture for spectral data classification.

    Attributes:
        causal_conv (CausalConv1d): Initial causal projection.
        blocks (nn.ModuleList): Stacked residual blocks.
        adaptive_pool (nn.AdaptiveMaxPool1d): Pooling layer.
        final_conv (nn.Conv1d): 1x1 convolution before pooling.
        fc_layers (nn.Sequential): Classification head.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2) -> None:
        """
        Initializes the WaveNet model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(WaveNet, self).__init__()

        self.causal_conv = CausalConv1d(1, 32, kernel_size=2)

        # Hyperparameters
        self.n_layers = 8
        self.n_blocks = 3
        self.channels = 32
        self.kernel_size = 2

        # Build the WaveNet blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            for i in range(self.n_layers):
                dilation = 2**i
                self.blocks.append(
                    ResidualBlock(
                        channels=self.channels,
                        kernel_size=self.kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                    )
                )

        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
        self.final_conv = nn.Conv1d(self.channels, self.channels, 1)
        self.relu = nn.ReLU()

        self.fc_layers = nn.Sequential(
            nn.Linear(self.channels * 4, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input spectrum of shape (B, D).

        Returns:
            torch.Tensor: Output logits.
        """
        # Ensure input has 3 dimensions [batch_size, seq_length, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.causal_conv(x)

        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        x = torch.stack(skip_connections).sum(dim=0)
        x = self.relu(x)

        x = self.final_conv(x)
        x = self.relu(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        x = self.fc_layers(x)

        return x

    def apply_weight_norm(self) -> None:
        """
        Applies weight normalization to all convolution and linear layers.
        """

        def _apply_weight_norm(module):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                weight_norm(module)

        self.apply(_apply_weight_norm)

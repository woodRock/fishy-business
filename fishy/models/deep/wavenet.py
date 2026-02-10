"""WaveNet model for time series classification.

This model uses causal convolutions and residual blocks to process sequential data.
It includes an initial causal convolutional layer, multiple residual blocks for feature extraction,
and a fully connected layer for classification. The architecture is designed to handle time series or other ordered
data, leveraging the strengths of convolutional neural networks for sequential tasks.

References:
1. van den Oord, A., Dieleman, S., Zen, H.,
   Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016).
   WaveNet: A generative model for raw audio.
   arXiv preprint arXiv:1609.03499.
2. Srivastava, N., Hinton, G., Krizhevsky, A.,
   Sutskever, I., & Salakhutdinov, R. (2014).
   Dropout: a simple way to prevent neural networks from overfitting.
   The journal of machine learning research, 15(1), 1929-1958.
3. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever,
   I., & Salakhutdinov, R. R. (2012).
   Improving neural networks by preventing co-adaptation of feature detectors.
   arXiv preprint arXiv:1207.0580.
4. Loshchilov, I., & Hutter, F. (2017).
   Decoupled weight decay regularization.
   arXiv preprint arXiv:1711.05101.
5. Hendrycks, D., & Gimpel, K. (2016).
   Gaussian error linear units (gelus).
   arXiv preprint arXiv:1606.08415.
6. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
   Rethinking the inception architecture for computer vision.
   In Proceedings of the IEEE conference on computer vision
   and pattern recognition (pp. 2818-2826).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with dilations
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, **kwargs
    ) -> None:
        """Initialize the CausalConv1d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            dilation (int): Dilation factor for the convolution. Defaults to 1.
            **kwargs: Additional arguments for the convolution layer.
        """
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        # Remove weight normalization from initialization
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs
        )

    def forward(self, x):
        """Forward pass through the CausalConv1d layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length - padding).
        """
        x = self.conv(x)
        if self.padding != 0:
            return x[:, :, : -self.padding]
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections and gated activation
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.2) -> None:
        """Initialize the ResidualBlock.

        Args:
            channels (int): Number of input and output channels.
            kernel_size (int): Size of the convolution kernel.
            dilation (int): Dilation factor for the convolution.
            dropout (float): Dropout rate for regularization. Defaults to 0.2.
        """
        super(ResidualBlock, self).__init__()

        self.dilated_conv = CausalConv1d(
            channels, 2 * channels, kernel_size, dilation=dilation
        )
        # Remove weight normalization from initialization
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass through the ResidualBlock.

        This block applies a dilated convolution, followed by a gated activation unit,
        a 1x1 convolution, and finally adds the input to the output (residual connection).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, sequence_length),
            where channels is the number of input channels.
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
    """WaveNet model for time series classification.
    This model uses causal convolutions and residual blocks to process sequential data.
    It includes an initial causal convolutional layer, multiple residual blocks for feature extraction,
    and a fully connected layer for classification. The architecture is designed to handle time series or other ordered
    data, leveraging the strengths of convolutional neural networks for sequential tasks.
    """

    def __init__(self, input_dim, output_dim, dropout=0.2) -> None:
        """Initialize the WaveNet model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            dropout (float): Dropout rate for regularization. Defaults to 0.2.
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
        # Remove weight normalization from initialization
        self.final_conv = nn.Conv1d(self.channels, self.channels, 1)
        self.relu = nn.ReLU()

        self.fc_layers = nn.Sequential(
            nn.Linear(self.channels * 4, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        """Forward pass through the WaveNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim),
            where output_dim is the number of classes.
        """
        # Ensure input has 3 dimensions [batch_size, seq_length, features]
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

    def apply_weight_norm(self):
        """
        Apply weight normalization to all convolution layers after model initialization
        """

        def _apply_weight_norm(module):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                weight_norm(module)

        self.apply(_apply_weight_norm)

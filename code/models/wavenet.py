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
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        # Remove weight normalization from initialization
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation, **kwargs
        )
        
    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            return x[:, :, :-self.padding]
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with skip connections and gated activation
    """
    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        self.dilated_conv = CausalConv1d(
            channels, 2 * channels, kernel_size, 
            dilation=dilation
        )
        # Remove weight normalization from initialization
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
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
    def __init__(self, input_dim, output_dim, dropout=0.2):
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
                dilation = 2 ** i
                self.blocks.append(
                    ResidualBlock(
                        channels=self.channels,
                        kernel_size=self.kernel_size,
                        dilation=dilation,
                        dropout=dropout
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
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
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
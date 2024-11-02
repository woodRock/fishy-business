import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # Remove the incorrect LayerNorm implementation
        # Instead, use normalization after the convolution
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Use 1D batch normalization instead of layer normalization
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.bn2 = nn.BatchNorm1d(n_outputs)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight.data)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(TCN, self).__init__()
        
        # Modified architecture with better stability
        num_channels = [32, 64, 64]  # Reduced complexity
        kernel_size = 3
        
        # Remove the problematic input_norm layer and use BatchNorm1d instead
        self.input_norm = nn.BatchNorm1d(1)  # For normalizing the channel dimension
        
        self.tcn = TemporalConvNet(
            num_inputs=1,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
        self.flatten = nn.Flatten()
        
        # Added batch normalization and modified FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(num_channels[-1] * 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_dim)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: [batch_size, sequence_length]
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, sequence_length]
        x = self.input_norm(x)  # Normalize along the channel dimension
        x = self.tcn(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
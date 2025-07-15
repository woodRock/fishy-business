"""Convolutional Neural Network for classification.

References:
1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
    Gradient-based learning applied to document recognition.
    Proceedings of the IEEE, 86(11), 2278-2324.
2. LeCun, Y. (1989).
    Generalization and network design strategies.
    Connectionism in perspective, 19(143-155), 18.
3. LeCun, Y., Boser, B., Denker, J. S., Henderson, D.,
    Howard, R. E., Hubbard, W., & Jackel, L. D. (1989).
    Backpropagation applied to handwritten zip code recognition.
    Neural computation, 1(4), 541-551.
4. LeCun, Y., Boser, B., Denker, J., Henderson, D.,
    Howard, R., Hubbard, W., & Jackel, L. (1989).
    Handwritten digit recognition with a back-propagation network.
    Advances in neural information processing systems, 2.
5. Srivastava, N., Hinton, G., Krizhevsky, A.,
    Sutskever, I., & Salakhutdinov, R. (2014).
    Dropout: a simple way to prevent neural networks from overfitting.
    The journal of machine learning research, 15(1), 1929-1958.
6. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever,
    I., & Salakhutdinov, R. R. (2012).
    Improving neural networks by preventing co-adaptation of feature detectors.
    arXiv preprint arXiv:1207.0580.
7. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
    Rethinking the inception architecture for computer vision.
    In Proceedings of the IEEE conference on computer vision
    and pattern recognition (pp. 2818-2826).
8. Hendrycks, D., & Gimpel, K. (2016).
    Gaussian error linear units (gelus).
    arXiv preprint arXiv:1606.08415.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout)

        # Shortcut path: if in_channels and out_channels differ, adjust with a 1x1 conv
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)  # Match dimensions
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += residual  # Add the shortcut (residual) connection
        out = self.relu(out)
        return out


class RCNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.5):
        super(RCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            ResidualBlock(1, 32, dropout=dropout),  # First block expects 1 channel
            ResidualBlock(32, 64, dropout=dropout, downsample=True),  # Downsample here
            ResidualBlock(64, 128, dropout=dropout),
            ResidualBlock(
                128, 256, dropout=dropout, downsample=True
            ),  # Downsample here
            nn.AdaptiveMaxPool1d(4),  # Fixed output size to 4
        )

        self.flatten = nn.Flatten()
        self.flat_features = 256 * 4  # Adjusted based on the pooling layer

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

"""Convolutional Neural Network for classification.

This module implements a Convolutional Neural Network (CNN) for classification tasks.
It includes convolutional layers, batch normalization, dropout, and fully connected layers.
The architecture is designed to handle 1D input data, such as time series or sequential data.
It uses ReLU activation and dropout for regularization.

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


class RCNN(nn.Module):
    """Residual Convolutional Neural Network (RCNN) for classification.
    This model consists of multiple residual blocks followed by fully connected layers.
    It includes batch normalization, ReLU activation, and dropout for regularization.
    The architecture is designed to handle 1D input data, such as time series or sequential data.
    """

    def __init__(self, input_dim: int = 1023, output_dim: int = 7, dropout: float = 0.5) -> None:
        """Initialize the RCNN model.

        Args:
            input_dim (int): Size of the input features.
            output_dim (int): Number of output classes.
            dropout (float): Dropout rate for regularization. Defaults to 0.5.
        """
        super(RCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        
        # Calculate the size of the flattened features after convolutions
        self.flat_features = 256 * (input_dim // 4)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        """Forward pass through the RCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes),
            where num_classes is the number of output classes.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


__all__ = ["RCNN"]
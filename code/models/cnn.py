""" Convolutional Neural Network for classification.

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


class CNN(nn.Module):
    def __init__(
        self, input_size: int = 1023, num_classes: int = 7, dropout: int = 0.5
    ) -> None:

        super(CNN, self).__init__()

        # Convolutional neural network (LeCun 1989,1989,1998)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),  # Batch normalization
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),  # Batch normalization
            # GELU activation (Hendrycks 2016)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()

        # Calculate the size of the flattened features after convolutions
        self.flat_features = 256 * (input_size // 4)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.ReLU(),
            # Dropout layer (Srivastava 2014, Hinton 2012)
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """Forward pass for the CNN.

        Args:
            x (torch.Tensor): the input tensor.

        Returns:
            x (torch.Tensor): the output tensor.
        """
        # Add channel dimension
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten the output
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc_layers(x)

        return x

__all__ = ["CNN"] # List of all classes in this module

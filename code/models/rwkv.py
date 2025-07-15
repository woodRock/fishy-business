""" Recurrent Weighted Key-Value (RWKV) model for time series classification.

This module implements a Recurrent Weighted Key-Value (RWKV) model for time series classification.
It includes linear layers for key, value, and output, and uses a recurrent mechanism to update
the hidden state based on the input features. The model is designed to handle sequential data
and is suitable for tasks such as time series forecasting or classification.    

References:
1. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., & Tegmark, M. (2024).
   Kan: Kolmogorov-arnold networks.
   arXiv preprint arXiv:2404.19756.
2. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
    Dropout: a simple way to prevent neural networks from overfitting.
    The journal of machine learning research, 15(1), 1929-1958.
3. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever,
    I., & Salakhutdinov, R. R. (2012).
    Improving neural networks by preventing co-adaptation of feature detectors.
    arXiv preprint arXiv:1207.0580.
4. Hendrycks, D., & Gimpel, K. (2016).
    Gaussian error linear units (gelus).
    arXiv preprint arXiv:1606.08415.
5. Loshchilov, I., & Hutter, F. (2017).
    Decoupled weight decay regularization.
    arXiv preprint arXiv:1711.05101.
6. Loshchilov, I., & Hutter, F. (2017).
    Decoupled weight decay regularization.
    arXiv preprint arXiv:1711.05101.
7. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
    Rethinking the inception architecture for computer vision.
    In Proceedings of the IEEE conference on computer vision
    and pattern recognition (pp. 2818-2826).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


class RWKV(nn.Module):
    """Receptance-Weighted Key-Value (RWKV) model for time series classification."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1) -> None:
        """Initialize the RWKV model.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in the recurrent layer.
            output_dim (int): Number of output classes.
            dropout (float): Dropout rate for regularization. Defaults to 0.1.
        """
        super(RWKV, self).__init__()
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_dim

        # Linear layers for key, value, and output
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)
        self.recurrent_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialization
        self.hidden = None

    def forward(self, x):
        """Forward pass through the RWKV model. 

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).    

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim),
            where output_dim is the number of classes.
        """
        # Compute keys and values
        keys = self.key_layer(x)
        values = self.value_layer(x)

        # If hidden state is not initialized, initialize it
        if self.hidden is None:
            self.hidden = torch.zeros(x.size(0), self.hidden_dim).to(x.device)

        # Update hidden state with current keys and values
        self.hidden = self.hidden + torch.tanh(keys + self.recurrent_layer(values))

        # Compute output
        output = self.output_layer(self.hidden)

        # Reset the hidden state.
        self.hidden = None
        return output

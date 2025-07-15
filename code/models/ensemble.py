""" Ensemble model combining multiple architectures.

This module implements an ensemble model that combines predictions from multiple architectures,
including Transformers and potentially other models like LSTM or Mamba.
The ensemble uses weighted voting to aggregate predictions from the individual models.

References:
1. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
   Dropout: a simple way to prevent neural networks from overfitting.
   The journal of machine learning research, 15(1), 1929-1958   
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
   Attention is all you need. In Advances in neural information processing systems (pp. 6000-6010).
3. Choromanski, K., Likhosherstov, V., Dohan, D., Gane, A., & Kaiser, Ł. (2021).
   Mamba: Efficient attention with linear memory and time complexity.
   In International Conference on Learning Representations (ICLR).
4. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., & Lin, S. (2021).
   Swin transformer: Hierarchical vision transformer using shifted windows.
   In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 10012-10022).
5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zoller, D., & Anselmi, F. (2020).
   An image is worth 16x16 words: Transformers for image recognition at scale.
   In International Conference on Learning Representations (ICLR).
"""
import torch
import torch.nn as nn
from .transformer import Transformer
from .lstm import LSTM
from .mamba import Mamba


class Ensemble(nn.Module):
    """Simple stacked voting classifier combining LSTM, Transformer, and Mamba."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """ Initialize the ensemble model.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Hidden dimension for the Transformer and Mamba models.
            output_dim (int): Number of output classes.
            dropout (float): Dropout rate for regularization.
            device (str): Device to run the model on (default: "cuda" if available, else "cpu").
        """
        super().__init__()
        self.device = device

        # Base models
        # self.lstm = LSTM(
        #     input_size=input_dim,
        #     hidden_size=hidden_dim,
        #     num_layers=4,
        #     output_size=output_dim,
        #     dropout=dropout,
        # )

        self.t1 = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=2,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.t2 = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=4,
            hidden_dim=hidden_dim,
            num_layers=4,
            dropout=dropout,
        )

        self.t3 = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=8,
            hidden_dim=hidden_dim,
            num_layers=8,
            dropout=dropout,
        )

        # self.mamba = Mamba(
        #     d_model=input_dim,
        #     d_state=hidden_dim,
        #     d_conv=4,
        #     expand=2,
        #     depth=4,
        #     n_classes=output_dim,
        #     dropout=dropout,
        # )

        # Voting weights
        self.voting_weights = nn.Parameter(torch.ones(3) / 3)

        # Move to device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining predictions from all models.
        
        Args: 
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim),
            where output_dim is the number of classes.
        """
        x = x.to(self.device)

        # Get predictions from each model
        t1 = self.t1(x)
        t2 = self.t2(x)
        t3 = self.t3(x)

        # Weighted voting
        weighted_sum = (
            self.voting_weights[0] * t1
            + self.voting_weights[1] * t2
            + self.voting_weights[2] * t3
        )

        return weighted_sum


__all__ = ["Ensemble"]

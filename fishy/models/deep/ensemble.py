import torch
import torch.nn as nn

from .transformer import Transformer


class Ensemble(nn.Module):
    """
    Simple averaging ensemble of Transformer models with different complexities.

    Attributes:
        t1 (Transformer): Shallow transformer (2 layers, 2 heads).
        t2 (Transformer): Medium transformer (4 layers, 4 heads).
        t3 (Transformer): Deep transformer (8 layers, 8 heads).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        """
        Initializes the ensemble model.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Hidden dimension for the backbone Transformers.
            output_dim (int): Number of output classes.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super().__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining predictions from all models.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Averaged output logits of shape (batch_size, output_dim).
        """
        # Get predictions from each model
        t1_out = self.t1(x)
        t2_out = self.t2(x)
        t3_out = self.t3(x)

        # Average the outputs
        return (t1_out + t2_out + t3_out) / 3.0

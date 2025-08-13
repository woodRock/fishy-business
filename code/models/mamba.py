"""Mamba model implementation.

This module implements the Mamba model, a linear-time sequence modeling architecture
with selective state spaces, as described in the paper by Gu and Dao (2023).
It includes the Mamba block and the overall Mamba model class.

References:
1. Gu, A., & Dao, T. (2023).
    Mamba: Linear-time sequence modeling with selective state spaces.
    arXiv preprint arXiv:2312.00752.
2. Srivastava, N., Hinton, G., Krizhevsky, A.,
    Sutskever, I., & Salakhutdinov, R. (2014).
    Dropout: a simple way to prevent neural networks from overfitting.
    The journal of machine learning research, 15(1), 1929-1958.
3. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever,
    I., & Salakhutdinov, R. R. (2012).
    Improving neural networks by preventing co-adaptation of feature detectors.
    arXiv preprint arXiv:1207.0580.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on computer
    vision and pattern recognition (pp. 770-778).
5. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).
    Layer normalization.
    arXiv preprint arXiv:1607.06450.
6. LeCun, Y. (1989).
    Generalization and network design strategies.
    Connectionism in perspective, 19(143-155), 18.
7. LeCun, Y., Boser, B., Denker, J., Henderson, D., Howard,
    R., Hubbard, W., & Jackel, L. (1989).
    Handwritten digit recognition with a back-propagation network.
    Advances in neural information processing systems, 2.
8. LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E.,
    Hubbard, W., & Jackel, L. D. (1989).
    Backpropagation applied to handwritten zip code recognition.
    Neural computation, 1(4), 541-551.
9. Hendrycks, D., & Gimpel, K. (2016).
    Gaussian error linear units (gelus).
    arXiv preprint arXiv:1606.08415.
10. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
    Rethinking the inception architecture for computer vision.
    In Proceedings of the IEEE conference on computer vision
    and pattern recognition (pp. 2818-2826).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        """Mamba block

        This block implements the inner and outer functions of the Mamba model.

        Args:
            d_model (int): the dimensions of the model.
            d_state (int): the dimensions of the state.
            d_conv (int): the dimensions of the convolution.
            expand (int): the expansion factor.
            dropout (float): the dropout rate.
            layer_norm_eps (float): the layer normalization epsilon.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.in_proj = nn.Linear(d_model, expand * d_model)
        # Convolutional layer (LeCun 1989, 1989, 1989)
        self.conv = nn.Conv1d(
            expand * d_model,
            expand * d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=expand * d_model,
        )
        # SiLU activation (Hendrycks 2016)
        self.activation = nn.SiLU()
        # Dropout (Srivastava 2014, Hinton 2012)
        self.dropout = nn.Dropout(dropout)
        # Layer normalization (Ba 2016)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.x_proj = nn.Linear(expand * d_model, d_state + d_model)
        self.dt_proj = nn.Linear(expand * d_model, d_state)

        self.out_proj = nn.Linear(expand * d_model, d_model)
        self.us_proj = nn.Linear(d_state, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): the input tensor.

        Returns:
            torch.Tensor: the output tensor.
        """
        B, L, D = x.shape
        x = self.layer_norm(x)  # Layer normalization

        x_in = self.in_proj(x)
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_act = self.activation(x_conv)
        x_act = self.dropout(x_act)

        x_and_ds = self.x_proj(x_act)
        dt = self.dt_proj(x_act)
        x, ds = x_and_ds.split([self.d_model, self.d_state], dim=-1)
        dt = F.softplus(dt)

        us = torch.zeros(B, L, self.d_state, device=x.device)
        for i in range(L):
            us[:, i] = us[:, i - 1] if i > 0 else us[:, i]
            us = us * torch.exp(-dt) + ds * torch.exp(-dt / 2)

        us_projected = self.us_proj(us)

        x = x + us_projected
        x = self.dropout(x)

        x = self.out_proj(torch.cat([x, x_act[:, :, self.d_model :]], dim=-1))
        x = self.dropout(x)

        return x


class Mamba(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        depth: int,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-5,
        spectral_norm: bool = True,
    ):
        super().__init__()
        self.embedding_layer = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                MambaBlock(d_model, d_state, d_conv, expand, dropout, layer_norm_eps)
                for _ in range(depth)
            ]
        )
        # Dropout (Srivastava 2014, Hinton 2012)
        self.dropout = nn.Dropout(dropout)

        if spectral_norm:
            # No self.fc, so apply spectral norm to a dummy layer or remove if not needed
            pass  # Removed self.fc, so no spectral norm here

        # Layer normalization (Ba 2016)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x):
        """Forward pass

        Args:
            x (torch.Tensor): the input tensor.

        Returns:
            torch.Tensor: the output tensor.
        """
        x = self.embedding_layer(x)  # Apply embedding layer first
        x = x.unsqueeze(1).repeat(1, 100, 1)

        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
            x = self.dropout(x)

        x = self.layer_norm(x)
        return x[:, 0, :]


class SiameseMamba(nn.Module):
    """A Siamese network using Mamba as the backbone."""

    def __init__(self, mamba_model: Mamba):
        """Initializes the SiameseMamba model.

        Args:
            mamba_model (Mamba): An instance of the Mamba model.
        """
        super(SiameseMamba, self).__init__()
        self.mamba = mamba_model
        # The output of Mamba is d_model, so the linear layer should take d_model as input
        self.fc = nn.Linear(1, 1) # Output is a scalar distance, so input to FC is 1

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SiameseMamba.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The output of the Siamese network.
        """
        # Pass through embedding layer
        x1_embed = self.mamba.embedding_layer(x1)
        x2_embed = self.mamba.embedding_layer(x2)

        # Pass through Mamba blocks, but without the unsqueeze and repeat
        # and without the final layer_norm and x[:, 0, :]
        # We need to get the feature vector before the final classification head
        # Assuming the Mamba model's layers produce a feature vector at the end
        # For Mamba, the output of the layers is (batch_size, sequence_length, d_model)
        # and the original Mamba takes x[:, 0, :] as the final output.
        # So we will apply the layers and then take x[:, 0, :]
        
        # Apply Mamba blocks to x1
        x1_processed = x1_embed.unsqueeze(1).repeat(1, 100, 1) # Re-add the unsqueeze and repeat for Mamba's internal processing
        for layer in self.mamba.layers:
            residual = x1_processed
            x1_processed = layer(x1_processed)
            x1_processed = x1_processed + residual
            x1_processed = self.mamba.dropout(x1_processed)
        x1_features = self.mamba.layer_norm(x1_processed)[:, 0, :] # Extract features

        # Apply Mamba blocks to x2
        x2_processed = x2_embed.unsqueeze(1).repeat(1, 100, 1) # Re-add the unsqueeze and repeat for Mamba's internal processing
        for layer in self.mamba.layers:
            residual = x2_processed
            x2_processed = layer(x2_processed)
            x2_processed = x2_processed + residual
            x2_processed = self.mamba.dropout(x2_processed)
        x2_features = self.mamba.layer_norm(x2_processed)[:, 0, :] # Extract features

        distance = F.pairwise_distance(x1_features, x2_features)
        output = self.fc(distance.unsqueeze(-1))
        return output

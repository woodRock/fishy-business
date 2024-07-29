import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, 
        d_model: int, 
        d_state: int, 
        d_conv: int, 
        expand: int,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-5,
    ) -> None:
<<<<<<< HEAD
        """ Mamba block
        
        Args:   
            d_model (int): the dimensions of the model.
            d_state (int): the dimensions of the state.
            d_conv (int): the dimensions of the convolution.
            expand (int): the expansion factor.
            dropout (float): the dropout rate.
            layer_norm_eps (float): the layer normalization epsilon.
=======
        """ Mamba block as described in the paper.
        
        Args: 
            d_model: The input and output feature dimension.
            d_state: The state dimension.
            d_conv: The convolutional dimension.
            expand: The number of dimensions to expand.
            dropout: The dropout rate.
            layer_norm_eps: The epsilon value for layer normalization.
            weight_decay: The L2 regularization
>>>>>>> 1e5bf639793de07acee6aaa4da95df9566f29c94
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.in_proj = nn.Linear(d_model, expand * d_model)
        self.conv = nn.Conv1d(expand * d_model, expand * d_model, kernel_size=d_conv, padding=d_conv-1, groups=expand * d_model)
        self.activation = nn.SiLU()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.x_proj = nn.Linear(expand * d_model, d_state + d_model)
        self.dt_proj = nn.Linear(expand * d_model, d_state)
        
        self.out_proj = nn.Linear(expand * d_model, d_model)
        self.us_proj = nn.Linear(d_state, d_model)
    
        
<<<<<<< HEAD
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass
        
        Args:  
            x (torch.Tensor): the input tensor.

        Returns: 
            torch.Tensor: the output tensor.
=======
    def forward(self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass of the model.
        
        Args:
            x: The input tensor of shape (B, L, D).

        Returns: 
            The output tensor of shape (B, L, D). 
>>>>>>> 1e5bf639793de07acee6aaa4da95df9566f29c94
        """
        B, L, D = x.shape
        
        x = self.layer_norm(x)  # Layer normalization
        
        x_in = self.in_proj(x)
        x_conv = x_in.transpose(1,2)
        x_conv = self.conv(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1,2)
        x_act = self.activation(x_conv)
        x_act = self.dropout(x_act)
        
        x_and_ds = self.x_proj(x_act)
        dt = self.dt_proj(x_act)
        x, ds = x_and_ds.split([self.d_model, self.d_state], dim=-1)
        dt = F.softplus(dt)
        
        us = torch.zeros(B, L, self.d_state, device=x.device)
        for i in range(L):
            us[:, i] = us[:, i-1] if i > 0 else us[:, i]
            us = us * torch.exp(-dt) + ds * torch.exp(-dt / 2)
        
        us_projected = self.us_proj(us)
        
        x = x + us_projected
        x = self.dropout(x)
        
        x = self.out_proj(torch.cat([x, x_act[:, :, self.d_model:]], dim=-1))
        x = self.dropout(x)

        return x

class Mamba(nn.Module):
    def __init__(self, 
        d_model: int, 
        d_state: int, 
        d_conv: int, 
        expand: int, 
        depth: int, 
        n_classes=2,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-5,
        spectral_norm: bool = True
    ):
<<<<<<< HEAD
        """ Mamba model
        
        Args: 
            d_model (int): the dimensions of the model.
            d_state (int): the dimensions of the state.
            d_conv (int): the dimensions of the convolution.
            expand (int): the expansion factor.
            depth (int): the depth of the model.
            n_classes (int): the number of classes.
            dropout (float): the dropout rate.
            layer_norm_eps (float): the layer normalization epsilon.
            spectral_norm (bool): whether to apply spectral normalization.

        References: 
            1. Gu, A., & Dao, T. (2023). 
=======
        """ Mamba model as described in the paper.
        
        Args: 
            d_model: The input and output feature dimension.
            d_state: The state dimension.
            d_conv: The convolutional dimension.
            expand: The number of dimensions to expand.
            depth: The number of layers.
            n_classes: The number of classes.
            dropout: The dropout rate.
            layer_norm_eps: The epsilon value for layer normalization.
            weight_decay: The L2 regularization
            spectral_norm: Whether to apply spectral normalization to the final linear layer.

        References: 
            1. Gu, A., & Dao, T. (2023).
>>>>>>> 1e5bf639793de07acee6aaa4da95df9566f29c94
            Mamba: Linear-time sequence modeling with selective state spaces. 
            arXiv preprint arXiv:2312.00752.
        """
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand, dropout, layer_norm_eps, weight_decay) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)
        
        if spectral_norm:
            self.fc = nn.utils.spectral_norm(self.fc)
        
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
<<<<<<< HEAD
    def forward(self, x):
        """ Forward pass
        
        Args: 
            x (torch.Tensor): the input tensor.
        
        Returns: 
            torch.Tensor: the output tensor.
=======
    def forward(self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass of the model.
        
        Args:
            x: The input tensor of shape (B, L, D).

        Returns:
            The output tensor of shape (B, C).
>>>>>>> 1e5bf639793de07acee6aaa4da95df9566f29c94
        """
        x = x.unsqueeze(1).repeat(1, 100, 1)
        
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
            x = self.dropout(x)
        
        x = self.layer_norm(x)  # Final layer normalization
        x = self.fc(x[:, 0, :])
        return x
<<<<<<< HEAD
=======
    
    def get_l2_regularization(self):
        """ Compute the L2 regularization for the model."""
        l2_reg = 0.0
        for layer in self.layers:
            for param in layer.parameters():
                l2_reg += torch.norm(param, p=2)
        return layer.weight_decay * l2_reg
>>>>>>> 1e5bf639793de07acee6aaa4da95df9566f29c94

# Usage example:
# model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2, depth=4)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# 
# # During training:
# loss = criterion(model(x), y) + model.get_l2_regularization()
# loss.backward()
# optimizer.step()
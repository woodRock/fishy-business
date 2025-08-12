import torch
import torch.nn as nn
import torch.nn.functional as F

class Hybrid(nn.Module):
    """
    A Hybrid CNN-Transformer model for sequential data.

    Assumes input shape: (batch_size, sequence_length, num_features_per_step)
    For REIMS data, this would typically be (batch_size, 2080, 1).
    """
    def __init__(
        self,
        input_dim: int,  # This will be num_features_per_step (e.g., 1 for REIMS intensity)
        output_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        cnn_channels: list = [32, 64],
        cnn_kernel_size: int = 3,
        cnn_stride: int = 1,
        cnn_padding: int = 1,
        cnn_pool_kernel_size: int = 2,
        cnn_pool_stride: int = 4,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # CNN Feature Extractor
        cnn_layers = []
        in_channels = input_dim
        for out_channels in cnn_channels:
            cnn_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=cnn_kernel_size,
                    stride=cnn_stride,
                    padding=cnn_padding,
                )
            )
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(
                nn.MaxPool1d(
                    kernel_size=cnn_pool_kernel_size, stride=cnn_pool_stride
                )
            )
            in_channels = out_channels
        self.cnn_feature_extractor = nn.Sequential(*cnn_layers)

        # Calculate the effective input dimension for the Transformer
        # We need to pass a dummy tensor to calculate the output shape of CNN
        # Assuming a sequence length of 2080 for calculation
        dummy_input = torch.randn(1, input_dim, 2080) # (batch, features_per_step, seq_len)
        cnn_output_channels = self.cnn_feature_extractor(dummy_input).shape[1]
        
        # Transformer Encoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_output_channels,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True, # Input and output tensors are (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers
        )

        # Final classification head
        self.fc_out = nn.Linear(cnn_output_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has 3 dimensions [batch_size, seq_length, features_per_step]
        if x.dim() == 2:
            # Assuming input is (batch_size, sequence_length) and features_per_step is 1
            x = x.unsqueeze(-1) # (batch_size, sequence_length, 1)

        # Permute for Conv1d: (batch_size, features_per_step, sequence_length)
        x = x.permute(0, 2, 1)

        # Pass through CNN
        x = self.cnn_feature_extractor(x)

        # Permute back for Transformer: (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)

        # Pass through Transformer
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Final classification
        x = self.fc_out(x)
        return x

__all__ = ["Hybrid"]

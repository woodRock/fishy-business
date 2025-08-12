import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PerformerAttention(nn.Module):
    """Performer-style attention using Random Feature Attention (RFA)."""

    def __init__(
        self, input_dim: int, num_heads: int, num_random_features: int = 256
    ) -> None:
        """Initialize the PerformerAttention layer.

        Args:
            input_dim (int): Dimension of the input features to the attention layer (should be divisible by num_heads).
            num_heads (int): Number of attention heads.
            num_random_features (int): Number of random features for approximation.
        """
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.num_random_features = num_random_features

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        # Random features for approximation
        # Shape: (num_heads, head_dim, num_random_features)
        self.register_buffer(
            "random_features",
            torch.randn(num_heads, self.head_dim, num_random_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

        # Apply random features
        # (batch_size, num_heads, seq_len, num_random_features)
        q_prime = torch.exp(
            torch.matmul(q, self.random_features)
            - 0.5 * q.pow(2).sum(dim=-1, keepdim=True)
        )
        k_prime = torch.exp(
            torch.matmul(k, self.random_features)
            - 0.5 * k.pow(2).sum(dim=-1, keepdim=True)
        )

        # Compute approximate attention
        # (batch_size, num_heads, num_random_features, head_dim)
        kv_prime = torch.matmul(k_prime.transpose(-1, -2), v)
        # (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(q_prime, kv_prime)

        # Normalize (similar to softmax)
        # (batch_size, num_heads, seq_len, num_random_features)
        norm_factor = torch.matmul(
            q_prime, k_prime.sum(dim=-2, keepdim=True).transpose(-1, -2)
        )
        attn_output = attn_output / (norm_factor + 1e-6)

        attn_output = (
            attn_output.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.input_dim)
        )
        output = self.out_proj(attn_output)
        return output


class Performer(nn.Module):
    """Performer model for time series classification."""

    def __init__(
        self,
        input_dim: int,  # Raw input features per time step (e.g., 2080 for REIMS data)
        output_dim: int,
        num_heads: int,
        hidden_dim: int,  # Dimension of the internal representation and attention blocks
        num_layers: int = 1,
        dropout: float = 0.1,
        num_random_features: int = 256,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Project raw input_dim (e.g., 2080) to hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.attention_layers = nn.ModuleList(
            [
                PerformerAttention(hidden_dim, num_heads, num_random_features)
                for _ in range(num_layers)
            ]
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                hidden_dim, hidden_dim * 4
            ),  # Common practice for feed-forward size
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Ensure input has 3 dimensions [batch_size, seq_length, features_per_step]
        # Here, seq_length is 1, and features_per_step is 2080
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, 2080)

        # Project input features to hidden_dim
        # x shape: (batch_size, 1, 2080) -> (batch_size, 1, hidden_dim)
        x = self.input_projection(x)

        # Apply attention layers with residual connections
        for attention in self.attention_layers:
            residual = x
            x = self.layer_norm1(x)
            x = residual + self.dropout(attention(x))

        # Feed-forward with residual connection
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        # Global pooling and classification
        # Since seq_len is 1, this just removes the sequence dimension
        x = x.mean(dim=1)  # (batch_size, hidden_dim)

        # Final classification
        x = self.fc_out(x)
        return x


__all__ = ["Performer", "PerformerAttention"]

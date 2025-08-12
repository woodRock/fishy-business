import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerConfig


class Longformer(nn.Module):
    """
    A Longformer model for sequential data.

    Assumes input shape: (batch_size, sequence_length, num_features_per_step)
    For REIMS data, this would typically be (batch_size, 2080, 1).
    """

    def __init__(
        self,
        input_dim: int,  # This will be num_features_per_step (e.g., 1 for REIMS intensity)
        output_dim: int,
        num_heads: int,
        hidden_dim: int,  # Corresponds to Longformer's hidden_size
        num_layers: int = 1,  # Corresponds to Longformer's num_hidden_layers
        dropout: float = 0.1,
        attention_window: int = 128,  # Specific to Longformer
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Longformer Configuration
        config = LongformerConfig(
            attention_window=attention_window,
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_dim,  # *4, # Common practice for feed-forward size
            hidden_act="gelu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            # vocab_size is not directly relevant for continuous input, but required
            # Set to a dummy value or adjust if using tokenization
            vocab_size=2,  # Minimum vocab_size to satisfy nn.Embedding assertion
            pad_token_id=0,  # Must be within vocab_size
        )

        # If input_dim (features_per_step) is not equal to hidden_dim, project it
        self.input_projection = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.longformer = LongformerModel(config)

        # Final classification head
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has 3 dimensions [batch_size, sequence_length, features_per_step]
        if x.dim() == 2:
            # Assuming input is (batch_size, sequence_length) and features_per_step is 1
            x = x.unsqueeze(-1)  # (batch_size, sequence_length, 1)

        # Project input features to hidden_dim if necessary
        x = self.input_projection(x)

        # Create attention mask (all ones for full attention on valid sequence)
        attention_mask = torch.ones(x.shape[0], x.shape[1], device=x.device)

        # Pass through Longformer model
        # We pass inputs_embeds directly since our input is continuous numerical data
        outputs = self.longformer(inputs_embeds=x, attention_mask=attention_mask)

        # Get the last hidden state (output of the last layer)
        sequence_output = outputs.last_hidden_state

        # Global average pooling
        x = sequence_output.mean(dim=1)

        # Final classification
        x = self.fc_out(x)
        return x


__all__ = ["Longformer"]

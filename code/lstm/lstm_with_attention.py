import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    

    def forward(self, lstm_output):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class LSTM(nn.Module):
    def __init__(self, 
                 input_size: int = 1023, 
                 hidden_size: int = 128, 
                 num_layers: int = 2, 
                 output_size: int = 2,
                 dropout: float = 0.5
    ) -> None:
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size, 
                    hidden_size, 
                    batch_first=True) 
            for i in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) 
            for _ in range(num_layers)
        ])
        
        # Dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) 
            for _ in range(num_layers + 1)  # +1 for the final dropout before FC layer
        ])
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state and cell state
        h = [torch.zeros(1, batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(1, batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        
        # Process through LSTM layers with residual connections, layer normalization, and dropout
        for i in range(self.num_layers):
            lstm_out, (h[i], c[i]) = self.lstm_layers[i](x, (h[i], c[i]))
            lstm_out = self.layer_norms[i](lstm_out + x if i > 0 else lstm_out)
            x = self.dropouts[i](lstm_out)  # Apply dropout between LSTM layers
        
        # Apply attention mechanism
        context_vector, attention_weights = self.attention(x)
        
        # Apply final dropout before the fully connected layer
        context_vector = self.dropouts[-1](context_vector)
        
        # Decode the context vector
        out = self.fc(context_vector)
        return out
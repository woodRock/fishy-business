import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, 
                 input_size: int = 1023, 
                 hidden_size: int = 128, 
                 num_layers: int = 2, 
                 output_size: int = 2,
                 dropout: float = 0.2
    ) -> None:
        """Long-short term memory (LSTM) module

        Args: 
            input_size (int): the size of the input. Defaults to 1023.
            hidden_size (int): the dimensions of the hidden layer. Defaults to 128.
            num_layers (int): the number of hidden layers. Defaults to 2.

        References: 
            1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
        
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            batch_first=True)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, 
                x: torch.Tensor
        ) -> torch.Tensor:
        """ Forward pass of the LSTM
        
        Args: 
            x (torch.Tensor): the input to the model.

        Returns 
            out (torch.Tensor): the output of the model.
        """
        # x shape: (batch_size, sequence_length, input_size)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to unsqueeze the input to add a sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Dropout
        out = self.dropout(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
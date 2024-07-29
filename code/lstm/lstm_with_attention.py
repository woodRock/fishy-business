import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        """ Attention mechanism layer.
        
        Args: 
            hidden_size (int): Hidden size of the LSTM layer
        """
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    

    def forward(self, lstm_output):
        """ Forward pass of the attention mechanism layer.
        
        Args: 
            lstm_output (torch.Tensor): Output of the LSTM layer

        Returns: 
            context_vector (torch.Tensor): Context vector
            attention_weights (torch.Tensor): Attention weights
        """
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
        """ LSTM model with attention mechanism.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units in the LSTM layer
            num_layers (int): Number of LSTM layers
            output_size (int): Number of output classes
            dropout (float): Dropout rate

        References: 
            1. Hochreiter, S., & Schmidhuber, J. (1997). 
                Long short-term memory. 
                Neural computation, 9(8), 1735-1780.
            2. Srivastava, N., Hinton, G., Krizhevsky, A.,
                Sutskever, I., & Salakhutdinov, R. (2014).
                Dropout: a simple way to prevent neural networks from overfitting.
                The journal of machine learning research, 15(1), 1929-1958.
            3. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever,
                I., & Salakhutdinov, R. R. (2012).
                Improving neural networks by preventing co-adaptation of feature detectors.
                arXiv preprint arXiv:1207.0580.
            4. Loshchilov, I., & Hutter, F. (2017). 
                Decoupled weight decay regularization. 
                arXiv preprint arXiv:1711.05101.
            5. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
                Rethinking the inception architecture for computer vision.
                In Proceedings of the IEEE conference on computer vision
                and pattern recognition (pp. 2818-2826).
            6. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
                A. N., ... & Polosukhin, I. (2017).
                Attention is all you need.
                Advances in neural information processing systems, 30.
            7. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).
                Layer normalization. 
                arXiv preprint arXiv:1607.06450.
            8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). 
                Deep residual learning for image recognition. 
                In Proceedings of the IEEE conference on 
                computer vision and pattern recognition (pp. 770-778).
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers (Hochreiter 1997) with residual connections (He 2016)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size, 
                    hidden_size, 
                    batch_first=True) 
            for i in range(num_layers)
        ])
        
        # Layer normalization (Ba 2016)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) 
            for _ in range(num_layers)
        ])
        
        # Dropout layers (Srivastava 2014, Hinton 2012)
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) 
            for _ in range(num_layers + 1)  # +1 for the final dropout before FC layer
        ])
        
        # Attention mechanism (Vaswani 2017)
        self.attention = AttentionLayer(hidden_size)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the LSTM model.
        
        Args: 
            x (torch.Tensor): Input tensor

        Returns: 
            torch.Tensor: Output tensor
        """
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
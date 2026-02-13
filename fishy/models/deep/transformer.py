# -*- coding: utf-8 -*-
"""
Transformer model for time series classification.

This model uses multi-head attention and feed-forward layers to process sequential data.
It is designed to handle variable-length sequences and can be used for tasks such as classification or regression.
The architecture includes layer normalization, dropout for regularization, and a final fully connected layer for output.

References:

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
    A. N., ... & Polosukhin, I. (2017).
    Attention is all you need.
    Advances in neural information processing systems, 30.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on
    computer vision and pattern recognition (pp. 770-778).
3. LeCun, Y. (1989). Generalization and network design strategies.
    Connectionism in perspective, 19(143-155), 18.
4. LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard,
    R., Hubbard, W., & Jackel, L. (1989).
    Handwritten digit recognition with a back-propagation network.
    Advances in neural information processing systems, 2.
5. LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E.,
    Hubbard, W., & Jackel, L. D. (1989).
    Backpropagation applied to handwritten zip code recognition.
    Neural computation, 1(4), 541-551.
6. Hendrycks, D., & Gimpel, K. (2016).
    Gaussian error linear units (gelus).
    arXiv preprint arXiv:1606.08415.
7. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).
    Layer normalization. arXiv preprint arXiv:1607.06450.
8. Srivastava, N., Hinton, G., Krizhevsky, A.,
    Sutskever, I., & Salakhutdinov, R. (2014).
    Dropout: a simple way to prevent neural networks from overfitting.
    The journal of machine learning research, 15(1), 1929-1958.
9. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever,
    I., & Salakhutdinov, R. R. (2012).
    Improving neural networks by preventing co-adaptation of feature detectors.
    arXiv preprint arXiv:1207.0580.
10. Glorot, X., & Bengio, Y. (2010, March).
    Understanding the difficulty of training deep feedforward neural networks.
    In Proceedings of the thirteenth international conference on artificial intelligence and statistics (pp. 249-256).
    JMLR Workshop and Conference Proceedings.
11. Loshchilov, I., & Hutter, F. (2017).
    Decoupled weight decay regularization.
    arXiv preprint arXiv:1711.05101.
12. Goodfellow, Ian, Yoshua Bengio, and Aaron Courville.
    Deep learning. MIT press, 2016.
13. Morgan, N., & Bourlard, H. (1989).
    Generalization and parameter estimation in feedforward nets:
    Some experiments. Advances in neural information processing systems, 2.
14. Xiong, R., Yang, Y., He, D., Zheng, K.,
    Zheng, S., Xing, C., ... & Liu, T. (2020, November).
    On layer normalization in the transformer architecture.
    In International Conference on Machine Learning (pp. 10524-10533). PMLR.
15. Karpathy, Andrej (2023)
    Let's build GPT: from scratch, in code, spelled out.
    YouTube https://youtu.be/kCc8FmEb1nY?si=1vM4DhyqsGKUSAdV
16. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
    Rethinking the inception architecture for computer vision.
    In Proceedings of the IEEE conference on computer vision
    and pattern recognition (pp. 2818-2826).
17. He, Kaiming, et al. "Delving deep into rectifiers:
    Surpassing human-level performance on imagenet classification."
    Proceedings of the IEEE international conference on computer vision. 2015.
18. Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013).
    Exact solutions to the nonlinear dynamics of learning in
    deep linear neural networks. arXiv preprint arXiv:1312.6120.
19. He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on
    computer vision and pattern recognition (pp. 770-778).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, List


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Examples:
        >>> import torch
        >>> attn = MultiHeadAttention(input_dim=16, num_heads=2)
        >>> x = torch.randn(4, 10, 16)
        >>> y = attn(x)
        >>> y.shape
        torch.Size([4, 10, 16])

    Attributes:
        input_dim (int): Number of input features.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        qkv (nn.Linear): Combined projection for Q, K, and V.
        fc_out (nn.Linear): Final output projection.
        scale (float): Scaling factor for dot-product attention.
    """

    def __init__(self, input_dim: int, num_heads: int) -> None:
        """
        Initializes the MultiHeadAttention layer.

        Args:
            input_dim (int): Number of input features.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        assert input_dim % num_heads == 0

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Combined projection for Q, K, V
        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)

        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            return_attention (bool): Whether to return attention weights.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, input_dim).
            torch.Tensor (optional): Attention weights of shape (batch_size, num_heads, seq_length, seq_length).
        """
        batch_size = x.shape[0]

        # Single matrix multiplication for all projections
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Combine heads
        out = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.input_dim)
        out = self.fc_out(out)
        
        if return_attention:
            return out, attn
        return out


class Transformer(nn.Module):
    """
    Transformer architecture for 1D spectral data.

    Examples:
        >>> import torch
        >>> model = Transformer(input_dim=10, output_dim=2, num_heads=2, hidden_dim=32)
        >>> x = torch.randn(8, 10)
        >>> output = model(x)
        >>> output.shape
        torch.Size([8, 2])

    Attributes:
        attention_layers (nn.ModuleList): List of multi-head attention layers.
        feed_forward (nn.Sequential): Position-wise feed-forward network.
        layer_norm1 (nn.LayerNorm): Norm layer before attention.
        layer_norm2 (nn.LayerNorm): Norm layer before feed-forward.
        dropout (nn.Dropout): Dropout layer.
        fc_out (nn.Linear): Final classification/regression head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_heads: int = 4,
        **kwargs,
    ) -> None:
        """
        Initializes the Transformer model.

        Args:
            input_dim (int): Number of input features (m/z bins).
            output_dim (int): Number of output classes/dimensions.
            hidden_dim (int, optional): Intermediate dimension for embeddings and feed-forward. Defaults to 128.
            num_layers (int, optional): Number of transformer blocks. Defaults to 1.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.1.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
        """
        super().__init__()

        # We treat each feature as a token in a sequence.
        # input_dim here is the number of features.
        self.n_features = input_dim
        self.embedding_dim = hidden_dim

        # Ensure embedding_dim is divisible by num_heads
        if self.embedding_dim % num_heads != 0:
            self.embedding_dim = (self.embedding_dim // num_heads) * num_heads
            if self.embedding_dim == 0:
                self.embedding_dim = num_heads

        self.input_projection = nn.Linear(1, self.embedding_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_dim, self.embedding_dim))

        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(self.embedding_dim, num_heads) for _ in range(num_layers)]
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

        self.layer_norm1 = nn.LayerNorm(self.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(self.embedding_dim, output_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input spectrum of shape (batch_size, input_dim).
            return_attention (bool): Whether to return attention weights from all layers.

        Returns:
            torch.Tensor: Logits/predictions of shape (batch_size, output_dim).
            List[torch.Tensor] (optional): List of attention weights from each layer.
        """
        # Ensure input has 3 dimensions [batch_size, n_features, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3 and x.size(1) == 1:
            # Handle (B, 1, F) by transposing to (B, F, 1)
            x = x.transpose(1, 2)
        
        # Project each feature to embedding_dim
        x = self.input_projection(x) # (B, n_features, embedding_dim)
        x = x + self.pos_embedding

        attentions = []
        # Apply attention layers with residual connections
        for attention in self.attention_layers:
            residual = x
            x = self.layer_norm1(x)
            if return_attention:
                attn_out, attn_weights = attention(x, return_attention=True)
                x = residual + self.dropout(attn_out)
                attentions.append(attn_weights)
            else:
                x = residual + self.dropout(attention(x))

        # Feed-forward with residual connection
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        # Global average pooling and classification
        x = x.mean(dim=1)  # Global average pooling over features
        x = self.fc_out(x)
        
        if return_attention:
            return x, attentions
        return x


__all__ = [
    "Transformer",
    "MultiHeadAttention",
]

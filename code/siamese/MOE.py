"""
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
4. LeCun, Y., Boser, B., Denker, J., Henderson, D., Howard,
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
14. Karpathy, Andrej (2023)
    Let's build GPT: from scratch, in code, spelled out.
    YouTube https://youtu.be/kCc8FmEb1nY?si=1vM4DhyqsGKUSAdV
15. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
    Rethinking the inception architecture for computer vision.
    In Proceedings of the IEEE conference on computer vision
    and pattern recognition (pp. 2818-2826).
16. He, Kaiming, et al. "Delving deep into rectifiers:
    Surpassing human-level performance on imagenet classification."
    Proceedings of the IEEE international conference on computer vision. 2015.
17. Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013).
    Exact solutions to the nonlinear dynamics of learning in
    deep linear neural networks. arXiv preprint arXiv:1312.6120.
18. 8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). 
    Deep residual learning for image recognition. 
    In Proceedings of the IEEE conference on 
    computer vision and pattern recognition (pp. 770-778).
19. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991).
    Adaptive mixtures of local experts. 
    Neural computation, 3(1), 79-87.
20. Kaiser, L., Gomez, A. N., Shazeer, N., Vaswani, A., Parmar, N., Jones, L., & Uszkoreit, J. (2017). 
    One model to learn them all. 
    arXiv preprint arXiv:1706.05137.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int) -> None:
        super().__init__()
        assert input_dim % num_heads == 0

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Combined projection for Q, K, V
        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)

        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.input_dim)
        x = self.fc_out(x)
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(input_dim, num_heads) for _ in range(num_layers)]
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Ensure input has 3 dimensions [batch_size, seq_length, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)

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
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc_out(x)
        return x


class ExpertLayer(nn.Module):
    """Individual expert neural network"""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with gating network"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        k: int = 2,  # Top-k experts to use
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Reshape input for expert processing
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        
        # Get gates
        gates = self.gate(x_flat)  # (batch_size * seq_len, num_experts)
        
        # Select top-k experts
        gate_scores, expert_indices = torch.topk(gates, self.k, dim=-1)
        gate_scores = F.softmax(gate_scores, dim=-1)  # Normalize gate scores
        
        # Compute expert outputs
        expert_outputs = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Create mask for current expert
            mask = (expert_indices == i).any(dim=-1)
            if mask.any():
                expert_outputs[mask] += expert(x_flat[mask])
        
        # Combine expert outputs with gate scores
        combined_output = torch.zeros_like(x_flat)
        for i in range(self.k):
            expert_idx = expert_indices[:, i]
            gate_score = gate_scores[:, i].unsqueeze(-1)
            combined_output += gate_score * torch.stack([
                self.experts[idx](x_flat[batch_idx:batch_idx+1])
                for batch_idx, idx in enumerate(expert_idx)
            ]).squeeze(1)
        
        # Reshape back to original dimensions
        return combined_output.view(batch_size, seq_len, d_model)


class MOE(nn.Module):
    """Transformer with Mixture of Experts replacing the standard feed-forward network"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int = 1,
        num_experts: int = 4,
        k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(input_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Replace feed-forward with MoE
        self.moe_layers = nn.ModuleList([
            MixtureOfExperts(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                k=k,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has 3 dimensions [batch_size, seq_length, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply attention and MoE layers with residual connections
        for attention, moe in zip(self.attention_layers, self.moe_layers):
            # Attention block
            residual = x
            x = self.layer_norm1(x)
            x = residual + self.dropout(attention(x))
            
            # MoE block
            residual = x
            x = self.layer_norm2(x)
            x = residual + self.dropout(moe(x))
        
        # Global pooling and classification
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc_out(x)
        return x


# Example usage:
if __name__ == "__main__":
    # Model parameters
    input_dim = 512
    output_dim = 10
    num_heads = 8
    hidden_dim = 2048
    num_layers = 2
    num_experts = 4
    k = 2
    
    # Create model
    model = MOE(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        k=k
    )
    
    # Example input
    batch_size = 32
    seq_length = 16
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [batch_size, output_dim]
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int) -> None:
        super().__init__()
        assert input_dim % num_heads == 0

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)
        self.scale = self.head_dim**-0.5
        
        # For attention re-weighting
        self.attn_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, iteration: int = 0) -> torch.Tensor:
        batch_size = x.shape[0]

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Dynamic attention scaling based on iteration
        scale = self.scale * (1 + self.attn_weight * iteration)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.input_dim)
        x = self.fc_out(x)
        return x, attn.mean(dim=1)  # Return attention scores for confidence

class TestTimeTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_iterations: int = 3,
        confidence_threshold: float = 0.95
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
        
        # Test-time compute parameters
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.confidence_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    # def forward(self, 
    #         x: torch.Tensor, 
    #         max_iterations: int = None, 
    #         confidence_threshold: float = None
    # ) -> tuple[torch.Tensor, int]:
    #     if x.dim() == 2:
    #         x = x.unsqueeze(1)

    #     max_iter = max_iterations if max_iterations is not None else self.max_iterations
    #     conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
    #     current_x = x
    #     best_output = None
    #     best_confidence = -float('inf')  # Track highest confidence
    #     iterations_used = 0
        
    #     for iteration in range(max_iter):
    #         x_iter = current_x
            
    #         # Apply attention layers
    #         attn_scores = []
    #         for attention in self.attention_layers:
    #             residual = x_iter
    #             x_iter = self.layer_norm1(x_iter)
    #             x_iter, attn = attention(x_iter, iteration)
    #             x_iter = residual + self.dropout(x_iter)
    #             attn_scores.append(attn)

    #         # Feed-forward
    #         residual = x_iter
    #         x_iter = self.layer_norm2(x_iter)
    #         x_iter = residual + self.dropout(self.feed_forward(x_iter))

    #         # Compute output and confidence
    #         output = self.fc_out(x_iter.mean(dim=1))
    #         confidence = self.confidence_predictor(x_iter.mean(dim=1)).mean()  # Mean across batch
            
    #         # Update best output if this confidence is higher
    #         if confidence > best_confidence:
    #             best_output = output
    #             best_confidence = confidence
            
    #         iterations_used = iteration + 1
            
    #         # Early exit if confident enough
    #         if confidence >= conf_thresh and iteration > 0:
    #             break
                
    #         # Update state for next iteration
    #         current_x = x_iter.detach() + x * 0.1
            
    #     return best_output

    def forward(self, 
            x: torch.Tensor, 
            max_iterations: int = None, 
            confidence_threshold: float = None
    ) -> tuple[torch.Tensor, int]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        current_x = x
        best_output = None
        best_confidence = -float('inf')
        iterations_used = 0
        
        for iteration in range(max_iter):
            x_iter = current_x
            
            attn_scores = []
            for attention in self.attention_layers:
                residual = x_iter
                x_iter = self.layer_norm1(x_iter)
                x_iter, attn = attention(x_iter, iteration)
                x_iter = residual + self.dropout(x_iter)
                attn_scores.append(attn)

            residual = x_iter
            x_iter = self.layer_norm2(x_iter)
            x_iter = residual + self.dropout(self.feed_forward(x_iter))

            output = self.fc_out(x_iter.mean(dim=1))
            confidence = F.softmax(output, dim=-1).max(dim=-1)[0].mean()  # Max prob across batch
            
            if confidence > best_confidence:
                best_output = output
                best_confidence = confidence
            
            iterations_used = iteration + 1
            
            if confidence >= conf_thresh and iteration > 0:
                break
                
            current_x = x_iter.detach() + x * 0.1
            
        return best_output

    def inference_with_budget(self, 
                            x: torch.Tensor, 
                            compute_budget: float) -> torch.Tensor:
        """Alternative inference mode with fixed compute budget"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Estimate iterations based on budget (simplified)
        max_iter = min(int(compute_budget * self.max_iterations), self.max_iterations)
        output, _ = self.forward(x, max_iterations=max_iter)
        return output

# Example usage
if __name__ == "__main__":
    # Model parameters
    input_dim = 512
    output_dim = 10
    num_heads = 8
    hidden_dim = 2048
    
    # Create model
    model = TestTimeTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=2,
        max_iterations=3
    )
    
    # Dummy input
    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    
    # Standard inference
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Budget-constrained inference
    output_budget = model.inference_with_budget(x, compute_budget=0.5)
    print(f"Budget output shape: {output_budget.shape}")
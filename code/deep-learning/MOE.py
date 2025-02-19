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
import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from util import create_data_module

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
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        k: int = 2,
        dropout: float = 0.1,
        use_majority_voting: bool = False
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.use_majority_voting = use_majority_voting
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Gating network (still used for tracking even in majority voting)
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Expert usage tracking
        self.expert_usage_counts = defaultdict(int)
        self.total_tokens = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        if self.use_majority_voting:
            # Get outputs from all experts
            expert_outputs = []
            for expert in self.experts:
                expert_out = expert(x_flat)
                expert_outputs.append(expert_out)
                
            # Track usage (in voting mode, all experts are used equally)
            self.total_tokens += x_flat.size(0)
            for i in range(self.num_experts):
                self.expert_usage_counts[i] += x_flat.size(0)
            
            # Average the expert outputs (soft voting)
            combined_output = torch.stack(expert_outputs).mean(dim=0)
            
        else:
            # Original top-k routing logic
            gates = self.gate(x_flat)
            gate_scores, expert_indices = torch.topk(gates, self.k, dim=-1)
            gate_scores = F.softmax(gate_scores, dim=-1)
            
            # Track expert usage
            for i in range(self.num_experts):
                self.expert_usage_counts[i] += torch.sum(expert_indices == i).item()
            self.total_tokens += expert_indices.numel()
            
            # Process with selected experts
            expert_outputs = torch.zeros_like(x_flat)
            for i, expert in enumerate(self.experts):
                mask = (expert_indices == i).any(dim=-1)
                if mask.any():
                    expert_outputs[mask] += expert(x_flat[mask])
            
            combined_output = torch.zeros_like(x_flat)
            for i in range(self.k):
                expert_idx = expert_indices[:, i]
                gate_score = gate_scores[:, i].unsqueeze(-1)
                combined_output += gate_score * torch.stack([
                    self.experts[idx](x_flat[batch_idx:batch_idx+1])
                    for batch_idx, idx in enumerate(expert_idx)
                ]).squeeze(1)
        
        return combined_output.view(batch_size, seq_len, d_model)
    
    def get_expert_utilization(self):
        total = sum(self.expert_usage_counts.values())
        if total == 0:
            return [0] * self.num_experts
        return [self.expert_usage_counts[i] / total for i in range(self.num_experts)]


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
        use_majority_voting: bool = False,
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
                dropout=dropout,
                use_majority_voting=use_majority_voting
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

def load_data(
    dataset: str = "species"
) -> (TensorDataset, list):
    """ 
    Load the specified dataset and scale the features using StandardScaler.

    Args: 
        dataset (str): The name of the dataset to load. One of ["species", "part", "oil", "cross-species"].

    Returns: 
        scaled_dataset (TensorDataset): The scaled dataset.
        targets (list): The target labels.
    """
    data_module = create_data_module(
        dataset_name=dataset,
        batch_size=32,
    )

    data_loader, _ = data_module.setup()
    dataset = data_loader.dataset
    features = torch.stack([sample[0] for sample in dataset])
    labels = torch.stack([sample[1] for sample in dataset])

    features_np = features.numpy()
    scaler = StandardScaler()
    scaled_features_np = scaler.fit_transform(features_np)
    scaled_features = torch.tensor(scaled_features_np, dtype=torch.float32)
    
    scaled_dataset = TensorDataset(scaled_features, labels)
    targets = [sample[1].argmax(dim=0) for sample in dataset]
    
    return scaled_dataset, targets

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> nn.Module:
    """Train the transformer model with quality score in loss function"""
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5)
    
    # Move model to device
    model.to(device)

    best_val_accuracy = -float('inf')
    best_model = None
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x)
            
            # Compute classification loss
            classification_loss = criterion(logits, torch.argmax(y, dim=1))
            
            # Combine classification loss and quality score
            loss = classification_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Print average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                logits = model(x)  # Beam search is used here
                classification_loss = criterion(logits, torch.argmax(y, dim=1))
                
                # Accumulate validation loss
                val_loss += classification_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == torch.argmax(y, dim=1)).sum().item()
        
        # Print validation results
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()

    if best_model is not None:
        # Early stopping. Load the best model
        model.load_state_dict(best_model)

    return model, best_val_accuracy

import optuna
from optuna.trial import Trial
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import numpy as np

def objective(trial: Trial, dataset, targets, device, n_folds=3):
    """Optuna objective function for hyperparameter optimization."""
    
    # Define hyperparameter search space
    config = {
        'input_dim': 2080,  # Fixed based on your data
        'output_dim': 7,   # Fixed for parts dataset
        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8, 16]),
        'hidden_dim': trial.suggest_int('hidden_dim', 128, 512, step=64),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'num_experts': trial.suggest_int('num_experts', 4, 8),
        'k': trial.suggest_int('k', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    }
    
    # Set up k-fold cross validation
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, targets)):
        # Create data loaders for this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=config['batch_size'],
            sampler=train_sampler
        )
        val_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=val_sampler
        )
        
        # Initialize model with trial hyperparameters
        model = MOE(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            num_heads=config['num_heads'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_experts=config['num_experts'],
            k=config['k'],
            dropout=config['dropout']
        )
        
        # Train model
        trained_model, val_accuracy = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=20,  # Fixed number of epochs for tuning
            learning_rate=config['learning_rate'],
            device=device
        )
        
        cv_scores.append(val_accuracy)
        
        # Report intermediate value
        trial.report(val_accuracy, fold)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(cv_scores)

def tune_hyperparameters(dataset, targets, device, n_trials=100):
    """Run hyperparameter optimization using Optuna."""
    
    # Create study object
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5
        )
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, dataset, targets, device),
        n_trials=n_trials,
        timeout=None  # No timeout
    )
    
    # Print optimization results
    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_trial.params

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare data
    dataset, targets = load_data(dataset="part")
    
    # Run hyperparameter tuning
    best_params = tune_hyperparameters(
        dataset=dataset,
        targets=targets,
        device=device,
        n_trials=100
    )

    print(f"Best parameters: {best_params}")
    
    # Save best parameters
    # torch.save(best_params, 'best_moe_params.pt')
    
    return best_params

if __name__ == "__main__":
    best_params = main()
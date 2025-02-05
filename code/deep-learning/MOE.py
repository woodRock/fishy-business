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

def compare_moe_variants(num_runs=5):
    """Compare original MoE with majority voting MoE across multiple runs using stratified k-fold CV."""
    
    n_classes = {"species": 2, "part": 7, "oil": 7, "cross-species": 3}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Store results for each variant
    results = {
        'original': {dataset: [] for dataset in n_classes.keys()},
        'majority_voting': {dataset: [] for dataset in n_classes.keys()},
        'expert_utilization': {
            'original': {dataset: [] for dataset in n_classes.keys()},
            'majority_voting': {dataset: [] for dataset in n_classes.keys()}
        }
    }
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Set random seed for reproducibility
        torch.manual_seed(run)
        np.random.seed(run)
        
        for dataset in n_classes.keys():
            print(f"\nProcessing dataset: {dataset}")
            
            # Load data
            scaled_dataset, targets = load_data(dataset=dataset)
            input_dim = scaled_dataset[0][0].shape[0]
            
            # Set number of folds based on dataset
            n_splits = 3 if dataset == "part" else 5
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run)
            
            # Store fold results for this run
            fold_results_original = []
            fold_results_voting = []
            fold_expert_util_original = []
            fold_expert_util_voting = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_dataset, targets)):
                print(f"Fold {fold + 1}/{n_splits}")
                
                # Create train/val datasets for this fold
                train_dataset = Subset(scaled_dataset, train_idx)
                val_dataset = Subset(scaled_dataset, val_idx)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Train and evaluate original MoE
                original_model = MOE(
                    input_dim=input_dim,
                    output_dim=n_classes[dataset],
                    num_heads=8,
                    hidden_dim=128,
                    num_layers=4,
                    num_experts=4,
                    k=2,
                    use_majority_voting=False
                )
                
                original_model, acc_original = train_model(
                    original_model,
                    train_loader,
                    val_loader,
                    num_epochs=100,
                    learning_rate=0.001,
                    device=device
                )
                
                # Get expert utilization for original model
                expert_util_original = []
                for layer in original_model.moe_layers:
                    expert_util_original.append(layer.get_expert_utilization())
                avg_util_original = np.mean(expert_util_original, axis=0)
                
                # Train and evaluate majority voting MoE
                voting_model = MOE(
                    input_dim=input_dim,
                    output_dim=n_classes[dataset],
                    num_heads=8,
                    hidden_dim=128,
                    num_layers=4,
                    num_experts=4,
                    k=2,
                    use_majority_voting=True
                )
                
                voting_model, acc_voting = train_model(
                    voting_model,
                    train_loader,
                    val_loader,
                    num_epochs=100,
                    learning_rate=0.001,
                    device=device
                )
                
                # Get expert utilization for voting model
                expert_util_voting = []
                for layer in voting_model.moe_layers:
                    expert_util_voting.append(layer.get_expert_utilization())
                avg_util_voting = np.mean(expert_util_voting, axis=0)
                
                # Store fold results
                fold_results_original.append(acc_original)
                fold_results_voting.append(acc_voting)
                fold_expert_util_original.append(avg_util_original)
                fold_expert_util_voting.append(avg_util_voting)
                
                print(f"Fold {fold + 1} Results:")
                print(f"Original MoE Accuracy: {acc_original:.2f}%")
                print(f"Majority Voting MoE Accuracy: {acc_voting:.2f}%")
            
            # Average results across folds for this run
            run_acc_original = np.mean(fold_results_original)
            run_acc_voting = np.mean(fold_results_voting)
            run_util_original = np.mean(fold_expert_util_original, axis=0)
            run_util_voting = np.mean(fold_expert_util_voting, axis=0)
            
            # Store run results
            results['original'][dataset].append(run_acc_original)
            results['majority_voting'][dataset].append(run_acc_voting)
            results['expert_utilization']['original'][dataset].append(run_util_original)
            results['expert_utilization']['majority_voting'][dataset].append(run_util_voting)
            
            print(f"\nRun {run + 1} Average Results for {dataset}:")
            print(f"Original MoE Accuracy: {run_acc_original:.2f}%")
            print(f"Majority Voting MoE Accuracy: {run_acc_voting:.2f}%")
    
    # Calculate final statistics
    final_results = {
        'original': {},
        'majority_voting': {},
        'expert_utilization': {
            'original': {},
            'majority_voting': {}
        }
    }
    
    for dataset in n_classes.keys():
        # Calculate mean and std of accuracies
        orig_acc = np.array(results['original'][dataset])
        vote_acc = np.array(results['majority_voting'][dataset])
        
        final_results['original'][dataset] = {
            'mean': np.mean(orig_acc),
            'std': np.std(orig_acc)
        }
        final_results['majority_voting'][dataset] = {
            'mean': np.mean(vote_acc),
            'std': np.std(vote_acc)
        }
        
        # Calculate mean expert utilization
        orig_util = np.mean(results['expert_utilization']['original'][dataset], axis=0)
        vote_util = np.mean(results['expert_utilization']['majority_voting'][dataset], axis=0)
        
        final_results['expert_utilization']['original'][dataset] = orig_util
        final_results['expert_utilization']['majority_voting'][dataset] = vote_util
    
    print("\nFinal Results Averaged Over All Runs:")
    for dataset in n_classes.keys():
        print(f"\n{dataset} Dataset:")
        print(f"Original MoE: {final_results['original'][dataset]['mean']:.2f}% ± {final_results['original'][dataset]['std']:.2f}%")
        print(f"Majority Voting MoE: {final_results['majority_voting'][dataset]['mean']:.2f}% ± {final_results['majority_voting'][dataset]['std']:.2f}%")
        print("\nExpert Utilization:")
        print("Original:", [f"{x:.3f}" for x in final_results['expert_utilization']['original'][dataset]])
        print("Majority:", [f"{x:.3f}" for x in final_results['expert_utilization']['majority_voting'][dataset]])
    
    return final_results

def compare_expert_counts(num_runs=5):
    """Compare model performance with different numbers of experts."""
    
    n_classes = {"species": 2, "part": 7, "oil": 7, "cross-species": 3}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_counts = [2, 4, 8, 16]
    
    # Store results for each expert count
    results = {
        count: {dataset: [] for dataset in n_classes.keys()}
        for count in expert_counts
    }
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Set random seed for reproducibility
        torch.manual_seed(run)
        np.random.seed(run)
        
        for dataset in n_classes.keys():
            print(f"\nProcessing dataset: {dataset}")
            
            # Load data
            scaled_dataset, targets = load_data(dataset=dataset)
            input_dim = scaled_dataset[0][0].shape[0]
            
            # Set number of folds based on dataset
            n_splits = 3 if dataset == "part" else 5
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run)
            
            # Store fold results for this run
            fold_results = {count: [] for count in expert_counts}
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_dataset, targets)):
                print(f"Fold {fold + 1}/{n_splits}")
                
                # Create train/val datasets for this fold
                train_dataset = Subset(scaled_dataset, train_idx)
                val_dataset = Subset(scaled_dataset, val_idx)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Test each expert count
                for num_experts in expert_counts:
                    print(f"\nTesting with {num_experts} experts")
                    
                    # Initialize model
                    model = MOE(
                        input_dim=input_dim,
                        output_dim=n_classes[dataset],
                        num_heads=8,
                        hidden_dim=128,
                        num_layers=4,
                        num_experts=num_experts,
                        k=min(2, num_experts),  # Ensure k doesn't exceed num_experts
                        use_majority_voting=False
                    )
                    
                    # Train and evaluate
                    model, accuracy = train_model(
                        model,
                        train_loader,
                        val_loader,
                        num_epochs=100,
                        learning_rate=0.001,
                        device=device
                    )
                    
                    # Store results
                    fold_results[num_experts].append(accuracy)
                    print(f"Accuracy with {num_experts} experts: {accuracy:.2f}%")
            
            # Average results across folds
            for num_experts in expert_counts:
                run_accuracy = np.mean(fold_results[num_experts])
                results[num_experts][dataset].append(run_accuracy)
    
    # Calculate final statistics
    final_results = {
        count: {dataset: {'mean': 0.0, 'std': 0.0} for dataset in n_classes.keys()}
        for count in expert_counts
    }
    
    print("\nFinal Results Averaged Over All Runs:")
    for dataset in n_classes.keys():
        print(f"\n{dataset} Dataset:")
        for num_experts in expert_counts:
            accuracies = np.array(results[num_experts][dataset])
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            final_results[num_experts][dataset]['mean'] = mean_acc
            final_results[num_experts][dataset]['std'] = std_acc
            
            print(f"{num_experts} experts: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    return final_results

def compare_topk_routing(num_runs=5):
    """Compare model performance with different numbers of experts."""
    
    n_classes = {"species": 2, "part": 7, "oil": 7, "cross-species": 3}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_k_routings = [1,2,4]
    
    # Store results for each expert count
    results = {
        count: {dataset: [] for dataset in n_classes.keys()}
        for count in top_k_routings
    }
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Set random seed for reproducibility
        torch.manual_seed(run)
        np.random.seed(run)
        
        for dataset in n_classes.keys():
            print(f"\nProcessing dataset: {dataset}")
            
            # Load data
            scaled_dataset, targets = load_data(dataset=dataset)
            input_dim = scaled_dataset[0][0].shape[0]
            
            # Set number of folds based on dataset
            n_splits = 3 if dataset == "part" else 5
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run)
            
            # Store fold results for this run
            fold_results = {count: [] for count in top_k_routings}
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_dataset, targets)):
                print(f"Fold {fold + 1}/{n_splits}")
                
                # Create train/val datasets for this fold
                train_dataset = Subset(scaled_dataset, train_idx)
                val_dataset = Subset(scaled_dataset, val_idx)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Test each expert count
                for k in top_k_routings:
                    print(f"\nTesting with k={k}")
                    
                    # Initialize model
                    model = MOE(
                        input_dim=input_dim,
                        output_dim=n_classes[dataset],
                        num_heads=8,
                        hidden_dim=128,
                        num_layers=4,
                        num_experts=4,
                        k=k,  # Ensure k doesn't exceed num_experts
                        use_majority_voting=False
                    )
                    
                    # Train and evaluate
                    model, accuracy = train_model(
                        model,
                        train_loader,
                        val_loader,
                        num_epochs=100,
                        learning_rate=0.001,
                        device=device
                    )
                    
                    # Store results
                    fold_results[k].append(accuracy)
                    print(f"Accuracy with k={k} {accuracy:.2f}%")
            
            # Average results across folds
            for k in top_k_routings:
                run_accuracy = np.mean(fold_results[k])
                results[k][dataset].append(run_accuracy)
    
    # Calculate final statistics
    final_results = {
        k: {dataset: {'mean': 0.0, 'std': 0.0} for dataset in n_classes.keys()}
        for k in top_k_routings
    }
    
    print("\nFinal Results Averaged Over All Runs:")
    for dataset in n_classes.keys():
        print(f"\n{dataset} Dataset:")
        for k in top_k_routings:
            accuracies = np.array(results[k][dataset])
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            final_results[k][dataset]['mean'] = mean_acc
            final_results[k][dataset]['std'] = std_acc
            
            print(f"k={k}: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    return 
    
def compare_layer_depths(num_runs=5):
    """Compare model performance with different numbers of experts."""
    
    n_classes = {"species": 2, "part": 7, "oil": 7, "cross-species": 3}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_depths = [2,4,8]
    
    # Store results for each expert count
    results = {
        count: {dataset: [] for dataset in n_classes.keys()}
        for count in layer_depths
    }
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Set random seed for reproducibility
        torch.manual_seed(run)
        np.random.seed(run)
        
        for dataset in n_classes.keys():
            print(f"\nProcessing dataset: {dataset}")
            
            # Load data
            scaled_dataset, targets = load_data(dataset=dataset)
            input_dim = scaled_dataset[0][0].shape[0]
            
            # Set number of folds based on dataset
            n_splits = 3 if dataset == "part" else 5
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run)
            
            # Store fold results for this run
            fold_results = {count: [] for count in layer_depths}
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_dataset, targets)):
                print(f"Fold {fold + 1}/{n_splits}")
                
                # Create train/val datasets for this fold
                train_dataset = Subset(scaled_dataset, train_idx)
                val_dataset = Subset(scaled_dataset, val_idx)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Test each expert count
                for depth in layer_depths:
                    print(f"\nTesting with depth={depth}")
                    
                    # Initialize model
                    model = MOE(
                        input_dim=input_dim,
                        output_dim=n_classes[dataset],
                        num_heads=8,
                        hidden_dim=128,
                        num_layers=depth,
                        num_experts=4,
                        k=2,  # Ensure k doesn't exceed num_experts
                        use_majority_voting=False
                    )
                    
                    # Train and evaluate
                    model, accuracy = train_model(
                        model,
                        train_loader,
                        val_loader,
                        num_epochs=100,
                        learning_rate=0.001,
                        device=device
                    )
                    
                    # Store results
                    fold_results[depth].append(accuracy)
                    print(f"Accuracy with depth={depth} {accuracy:.2f}%")
            
            # Average results across folds
            for depth in layer_depths:
                run_accuracy = np.mean(fold_results[depth])
                results[depth][dataset].append(run_accuracy)
    
    # Calculate final statistics
    final_results = {
        depth: {dataset: {'mean': 0.0, 'std': 0.0} for dataset in n_classes.keys()}
        for depth in layer_depths
    }
    
    print("\nFinal Results Averaged Over All Runs:")
    for dataset in n_classes.keys():
        print(f"\n{dataset} Dataset:")
        for depth in layer_depths:
            accuracies = np.array(results[depth][dataset])
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            final_results[depth][dataset]['mean'] = mean_acc
            final_results[depth][dataset]['std'] = std_acc
            
            print(f"depth={depth}: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    return final_results

def transfer_learning_experiment(num_runs=5):
    """
    Compare model performance with and without transfer learning.
    For each target dataset, trains:
    1. A model from scratch
    2. A model pre-trained on source dataset

    You should expect larger improvements when:

    1. Source and target tasks are more closely related
    2. Source dataset is larger than target dataset
    3. Target task has limited training data
    4. The underlying chemical signatures share common patterns

    Final Results Averaged Over All Runs:

    species->part:
    Baseline (No Transfer): 66.67% ± 0.00%
    With Transfer Learning: 66.67% ± 0.00%
    Improvement: 0.00%

    species->oil:
    Baseline (No Transfer): 50.00% ± 0.00%
    With Transfer Learning: 50.00% ± 0.00%
    Improvement: 0.00%

    part->cross-species:
    Baseline (No Transfer): 87.10% ± 0.00%
    With Transfer Learning: 80.65% ± 0.00%
    Improvement: -6.45%

    part->oil:
    Baseline (No Transfer): 46.15% ± 0.00%
    With Transfer Learning: 50.00% ± 0.00%
    Improvement: 3.85%

    oil->part:
    Baseline (No Transfer): 66.67% ± 0.00%
    With Transfer Learning: 58.33% ± 0.00%
    Improvement: -8.33%

    oil->cross-species:
    Baseline (No Transfer): 90.32% ± 0.00%
    With Transfer Learning: 80.65% ± 0.00%
    Improvement: -9.68%

    cross-species->part:
    Baseline (No Transfer): 66.67% ± 0.00%
    With Transfer Learning: 66.67% ± 0.00%
    Improvement: 0.00%

    cross-species->oil:
    Baseline (No Transfer): 57.69% ± 0.00%
    With Transfer Learning: 42.31% ± 0.00%
    Improvement: -15.38%
    """
    
    n_classes = {"species": 2, "part": 7, "oil": 7, "cross-species": 3}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transfer_pairs = [
        ("species", "oil"),
        ("species", "part"),
        ("part", "cross-species"),
        ("part", "oil"),
        ("oil", "part"),
        ("oil", "cross-species"),
        ("cross-species", "part"),
        ("cross-species", "oil")
    ]
    
    # Store results for each transfer pair
    results = {
        f"{src}->{tgt}": {
            "target_baseline": [],  # No pre-training
            "target_transfer": []   # With pre-training
        }
        for src, tgt in transfer_pairs
    }
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        torch.manual_seed(run)
        np.random.seed(run)
        
        for src_dataset, tgt_dataset in transfer_pairs:
            print(f"\nEvaluating: {src_dataset} -> {tgt_dataset}")
            
            # Load target dataset
            tgt_data, tgt_targets = load_data(tgt_dataset)
            n_splits = 3 if tgt_dataset == "part" else 5
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run)
            
            train_idx, val_idx = next(kfold.split(tgt_data, tgt_targets))
            tgt_train = Subset(tgt_data, train_idx)
            tgt_val = Subset(tgt_data, val_idx)
            
            tgt_train_loader = DataLoader(tgt_train, batch_size=32, shuffle=True)
            tgt_val_loader = DataLoader(tgt_val, batch_size=32, shuffle=False)
            
            # 1. Train model from scratch on target dataset (baseline)
            print("Training baseline model...")
            baseline_model = MOE(
                input_dim=2080,
                output_dim=n_classes[tgt_dataset],
                num_heads=8,
                hidden_dim=128,
                num_layers=4,
                num_experts=4,
                k=2
            ).to(device)
            
            _, baseline_accuracy = train_model(
                baseline_model,
                tgt_train_loader,
                tgt_val_loader,
                num_epochs=100,
                learning_rate=0.001,
                device=device
            )
            
            # 2. Train with transfer learning
            print("Training transfer learning model...")
            # First, train on source dataset
            transfer_model = MOE(
                input_dim=2080,
                output_dim=n_classes[src_dataset],
                num_heads=8,
                hidden_dim=128,
                num_layers=4,
                num_experts=4,
                k=2
            ).to(device)
            
            # Load and prepare source dataset
            src_data, src_targets = load_data(src_dataset)
            n_splits_src = 3 if src_dataset == "part" else 5
            src_kfold = StratifiedKFold(n_splits=n_splits_src, shuffle=True, random_state=run)
            
            src_train_idx, src_val_idx = next(src_kfold.split(src_data, src_targets))
            src_train = Subset(src_data, src_train_idx)
            src_val = Subset(src_data, src_val_idx)
            
            src_train_loader = DataLoader(src_train, batch_size=32, shuffle=True)
            src_val_loader = DataLoader(src_val, batch_size=32, shuffle=False)
            
            # Pre-train on source dataset
            transfer_model, _ = train_model(
                transfer_model,
                src_train_loader,
                src_val_loader,
                num_epochs=100,
                learning_rate=0.001,
                device=device
            )
            
            # Modify output layer for target dataset
            transfer_model.fc_out = nn.Linear(
                transfer_model.fc_out.in_features, 
                n_classes[tgt_dataset]
            ).to(device)
            
            # Fine-tune on target dataset
            _, transfer_accuracy = train_model(
                transfer_model,
                tgt_train_loader,
                tgt_val_loader,
                num_epochs=100,
                learning_rate=0.0001,  # Lower learning rate for fine-tuning
                device=device
            )
            
            # Store results
            pair_key = f"{src_dataset}->{tgt_dataset}"
            results[pair_key]["target_baseline"].append(baseline_accuracy)
            results[pair_key]["target_transfer"].append(transfer_accuracy)
            
            print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
            print(f"Transfer learning accuracy: {transfer_accuracy:.2f}%")
    
    # Calculate final statistics
    print("\nFinal Results Averaged Over All Runs:")
    final_results = {}
    
    for pair in transfer_pairs:
        pair_key = f"{pair[0]}->{pair[1]}"
        baseline_acc = np.array(results[pair_key]["target_baseline"])
        transfer_acc = np.array(results[pair_key]["target_transfer"])
        
        final_results[pair_key] = {
            "baseline": {"mean": np.mean(baseline_acc), "std": np.std(baseline_acc)},
            "transfer": {"mean": np.mean(transfer_acc), "std": np.std(transfer_acc)}
        }
        
        print(f"\n{pair_key}:")
        print(f"Baseline (No Transfer): {np.mean(baseline_acc):.2f}% ± {np.std(baseline_acc):.2f}%")
        print(f"With Transfer Learning: {np.mean(transfer_acc):.2f}% ± {np.std(transfer_acc):.2f}%")
        print(f"Improvement: {np.mean(transfer_acc) - np.mean(baseline_acc):.2f}%")
    
    return final_results

if __name__ == "__main__":
    final_results = transfer_learning_experiment(num_runs=30)
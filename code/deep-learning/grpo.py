import copy 
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

class GRPOMixtureOfExperts(nn.Module):
    """MoE layer with GRPO optimization capabilities"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        k: int = 2,
        dropout: float = 0.1,
        beta: float = 0.01,      # KL penalty coefficient
        epsilon: float = 0.2,    # Clipping parameter
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.beta = beta
        self.epsilon = epsilon
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Expert usage tracking
        self.expert_usage_counts = defaultdict(int)
        self.total_tokens = 0
        
        # Reference policy for KL divergence
        self.ref_policy = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        # Get gating logits
        gates = self.gate(x_flat)
        gate_scores, expert_indices = torch.topk(gates, self.k, dim=-1)
        gate_scores = F.softmax(gate_scores, dim=-1)
        
        # Track expert usage
        for i in range(self.num_experts):
            self.expert_usage_counts[i] += torch.sum(expert_indices == i).item()
        self.total_tokens += expert_indices.numel()
        
        # Process with selected experts
        combined_output = torch.zeros_like(x_flat)
        for i in range(self.k):
            expert_idx = expert_indices[:, i]
            gate_score = gate_scores[:, i].unsqueeze(-1)
            combined_output += gate_score * torch.stack([
                self.experts[idx](x_flat[batch_idx:batch_idx+1])
                for batch_idx, idx in enumerate(expert_idx)
            ]).squeeze(1)
        
        return combined_output.view(batch_size, seq_len, d_model)
    
    def store_ref_policy(self):
        """Store current policy as reference for KL divergence computation"""
        self.ref_policy = copy.deepcopy(self)
        for param in self.ref_policy.parameters():
            param.requires_grad = False
    
    def get_policy_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over experts"""
        gates = self.gate(x.view(-1, x.size(-1)))
        return F.softmax(gates, dim=-1)
    
    def compute_kl_divergence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy
        DKL(πθ||πref) = πref(o|q)/πθ(o|q) - log(πref(o|q)/πθ(o|q)) - 1
        """
        if self.ref_policy is None:
            return torch.tensor(0.0, device=x.device)
            
        with torch.no_grad():
            ref_dist = self.ref_policy.get_policy_distribution(x)
        curr_dist = self.get_policy_distribution(x)
        
        # Add small epsilon to avoid division by zero
        ratio = (ref_dist / (curr_dist + 1e-8))
        kl_div = ratio - torch.log(ratio + 1e-8) - 1
        return kl_div.mean()

class GRPOTransformer(nn.Module):
    """Transformer with GRPO-enhanced MoE layers"""
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
        beta: float = 0.005,
        epsilon: float = 0.2,
        group_size: int = 32
    ):
        super().__init__()
        
        self.group_size = group_size
        self.beta = beta
        self.epsilon = epsilon
        
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(input_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.moe_layers = nn.ModuleList([
            GRPOMixtureOfExperts(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                k=k,
                dropout=dropout,
                beta=beta,
                epsilon=epsilon
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(input_dim, output_dim)
    
    def store_ref_policies(self):
        """Store reference policies for all MoE layers"""
        for layer in self.moe_layers:
            layer.store_ref_policy()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply attention and MoE layers with KL regularization
        kl_losses = []
        for attention, moe in zip(self.attention_layers, self.moe_layers):
            # Attention block
            residual = x
            x = self.layer_norm1(x)
            x = residual + self.dropout(attention(x))
            
            # MoE block with KL computation
            residual = x
            x = self.layer_norm2(x)
            moe_out = moe(x)
            x = residual + self.dropout(moe_out)
            
            # Compute KL divergence for this layer
            kl_losses.append(moe.compute_kl_divergence(x))
        
        # Average KL losses across layers
        self.kl_loss = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0)
        
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc_out(x)
        return x
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantages using group statistics"""
        rewards = rewards.view(-1, self.group_size)
        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = (rewards - mean_rewards) / std_rewards
        return advantages.view(-1)

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
            # Save the best model using deep copy.
            best_model = copy.deepcopy(model)

    if best_model is not None:
        # Early stopping. Load the best model
        model = best_model

    return model, best_val_accuracy

def compute_ms_rewards(
    outputs: torch.Tensor, 
    targets: torch.Tensor,  
    raw_features: torch.Tensor,
    confidence_threshold: float = 0.8
) -> torch.Tensor:
    """
    Compute comprehensive rewards for mass spectrometry predictions.
    
    Args:
        outputs: Model logits [batch_size, num_classes]
        targets: Ground truth labels [batch_size, num_classes]
        raw_features: Original MS features [batch_size, 2080]
        confidence_threshold: Threshold for high confidence predictions
    """
    device = outputs.device
    batch_size = outputs.size(0)
    rewards = torch.zeros(batch_size, device=device)
    
    # 1. Classification Accuracy (0-1)
    probs = F.softmax(outputs, dim=-1)
    predictions = torch.argmax(probs, dim=-1)
    true_labels = torch.argmax(targets, dim=-1)
    correct = (predictions == true_labels).float()
    rewards += correct * 0.4  # Base reward for correctness
    
    # 2. Prediction Confidence (0-0.3)
    max_probs = torch.max(probs, dim=-1)[0]
    confidence_reward = torch.where(
        correct == 1,
        max_probs - confidence_threshold,  # Reward confidence when correct
        confidence_threshold - max_probs    # Penalize confidence when wrong
    )
    rewards += 0.3 * confidence_reward
    
    # 3. Feature Importance Alignment (0-0.3)
    important_ranges = [
        (200, 300),   # Fatty acids region
        (700, 800),   # Phospholipids region
        (850, 950)    # Triglycerides region
    ]
    
    # Compute attention weights
    attention_weights = F.softmax(outputs @ torch.randn(outputs.size(1), raw_features.size(1), device=device), dim=-1)
    
    # Create a linspace for the m/z values
    linspace = torch.linspace(77.04, 999.32, steps=2080)
    
    range_rewards = []
    for start, end in important_ranges:
        start_idx = torch.argmin(torch.abs(linspace - start))
        end_idx = torch.argmin(torch.abs(linspace - end))
        range_attention = attention_weights[:, start_idx:end_idx].mean(dim=1)
        range_rewards.append(range_attention)
    
    feature_reward = torch.stack(range_rewards).mean(dim=0)
    rewards += 0.3 * feature_reward
    
    # 4. Peak Intensity Consistency
    class_mean_intensity = torch.stack([
        torch.mean(raw_features[true_labels == i], dim=0) 
        for i in range(outputs.size(1))
    ])
    peak_reward = torch.sum(attention_weights * class_mean_intensity[predictions], dim=1)
    rewards += 0.2 * peak_reward

    # 5. Signal-to-Noise Ratio
    snr = torch.mean(raw_features, dim=0) / torch.std(raw_features, dim=0)
    snr_reward = torch.sum(attention_weights * snr, dim=1)
    rewards += 0.2 * snr_reward

    return rewards

def train_with_grpo(
    model: GRPOTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device
) -> GRPOTransformer:
    """Train model using GRPO with KL regularization"""
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    
    # Initial storage of reference policies
    model.store_ref_policies()
    
    for epoch in range(num_epochs):
        model.train()
        running_stats = defaultdict(list)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            # Generate group of outputs and compute rewards
            all_logits = []
            all_rewards = []
            
            for _ in range(model.group_size):
                # Forward pass (includes KL computation)
                logits = model(x)
                all_logits.append(logits)
                
                # Compute accuracy reward
                predictions = torch.argmax(logits, dim=-1)
                rewards = compute_ms_rewards(logits, y, x)
                all_rewards.append(rewards)
            
            # Stack rewards and compute advantages
            rewards = torch.stack(all_rewards, dim=1)
            advantages = model.compute_advantages(rewards)
            
            # Get mean logits for policy ratio computation
            mean_logits = torch.stack(all_logits).mean(dim=0)
            policy_ratio = F.softmax(mean_logits, dim=-1).mean()
            
            # Compute losses
            surr1 = policy_ratio * advantages
            surr2 = torch.clamp(
                policy_ratio,
                1 - model.epsilon,
                1 + model.epsilon
            ) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Add KL penalty (computed during forward pass)
            total_loss = policy_loss + model.beta * model.kl_loss
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record statistics
            running_stats['policy_loss'].append(policy_loss.item())
            running_stats['kl_loss'].append(model.kl_loss.item())
            running_stats['total_loss'].append(total_loss.item())
            running_stats['reward'].append(rewards.mean().item())
            
            if (batch_idx + 1) % 10 == 0:
                stats_str = ', '.join(
                    f'{k}: {np.mean(v):.4f}'
                    for k, v in running_stats.items()
                )
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: {stats_str}")
        
        # Update reference policies periodically
        if (epoch + 1) % 5 == 0:
            model.store_ref_policies()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                predictions = torch.argmax(logits, dim=-1)
                total += y.size(0)
                correct += (predictions == torch.argmax(y, dim=-1)).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} Validation Accuracy: {val_accuracy:.2f}%")
        
    return model, val_accuracy

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


def main():
    """Main execution loop with supervised pretraining followed by GRPO fine-tuning"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    datasets = ["part", "oil", "cross-species"]
    n_classes = {"species": 2, "part": 7, "oil": 7, "cross-species": 3}
    results = defaultdict(list)
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        # Load data
        scaled_dataset, targets = load_data(dataset=dataset)
        input_dim = scaled_dataset[0][0].shape[0]
        output_dim = n_classes[dataset]
        
        # Setup cross validation
        n_splits = 3 if dataset == "part" else 5
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_accuracies = {
            'supervised': [],
            'grpo': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_dataset, targets)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            # Create data loaders
            train_dataset = Subset(scaled_dataset, train_idx)
            val_dataset = Subset(scaled_dataset, val_idx)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # 1. Supervised Pretraining
            print("\nSupervised Pretraining Phase:")
            pretrained_model = GRPOTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                num_heads=8,
                hidden_dim=128,
                num_layers=4,
                num_experts=4,
                k=2,
                beta=0.01,
                epsilon=0.2,
                group_size=16
            )
            
            pretrained_model, supervised_acc = train_model(
                model=pretrained_model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=50,  # Fewer epochs for pretraining
                learning_rate=0.001,
                device=device
            )
            
            print(f"Supervised Pretraining Accuracy: {supervised_acc:.2f}%")
            fold_accuracies['supervised'].append(supervised_acc)
            
            # 2. GRPO Fine-tuning
            print("\nGRPO Fine-tuning Phase:")
            grpo_model = GRPOTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                num_heads=8,
                hidden_dim=128,
                num_layers=4,
                num_experts=4,
                k=2,
                beta=0.005,
                epsilon=0.2,
                group_size=32
            )
            
            # Load pretrained weights
            grpo_model.load_state_dict(pretrained_model.state_dict(), strict=False)
            
            # Fine-tune with GRPO
            grpo_model, grpo_acc = train_with_grpo(
                model=grpo_model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=50,  # Fewer epochs for fine-tuning
                learning_rate=1e-5,  # Lower learning rate for fine-tuning
                device=device
            )
            
            print(f"GRPO Fine-tuning Accuracy: {grpo_acc:.2f}%")
            fold_accuracies['grpo'].append(grpo_acc)
        
        # Calculate average results
        supervised_mean = np.mean(fold_accuracies['supervised'])
        supervised_std = np.std(fold_accuracies['supervised'])
        grpo_mean = np.mean(fold_accuracies['grpo'])
        grpo_std = np.std(fold_accuracies['grpo'])
        
        results[dataset] = {
            'supervised': {'mean': supervised_mean, 'std': supervised_std},
            'grpo': {'mean': grpo_mean, 'std': grpo_std}
        }
        
        print(f"\n{dataset} Dataset Final Results:")
        print(f"Supervised: {supervised_mean:.2f}% ± {supervised_std:.2f}%")
        print(f"GRPO: {grpo_mean:.2f}% ± {grpo_std:.2f}%")
    
    # Print final summary
    print("\nFinal Results Summary:")
    print("=" * 50)
    for dataset, metrics in results.items():
        print(f"\n{dataset}:")
        print(f"Supervised: {metrics['supervised']['mean']:.2f}% ± {metrics['supervised']['std']:.2f}%")
        print(f"GRPO: {metrics['grpo']['mean']:.2f}% ± {metrics['grpo']['std']:.2f}%")
        print(f"Improvement: {metrics['grpo']['mean'] - metrics['supervised']['mean']:.2f}%")

if __name__ == "__main__":
    main()
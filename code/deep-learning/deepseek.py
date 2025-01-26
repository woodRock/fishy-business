"""
DeepSeek: Reinforcement Learning for Transformer-based Classification. 

A PyTorch implementation of a vanilla transformer-based classifier (Vaswani et al., 2017) 
trained using the Generalized Reward-Per-Optimization (GRPO) algorithm (Guo et al., 2025). 
The model is pre-trained using cross-entropy loss and fine-tuned using GRPO to incentivize reasoning capability in large language models (LLMs). 
The code is adapted from the AlphaGo Zero algorithm (Silver et al., 2017) and is designed to work with a variety of datasets.

Results: 

## Species 

Pretrain Validation Accuracy: Mean = 0.935, Std = 0.037
GRPO Validation Accuracy: Mean = 0.963, Std = 0.034

## Part 

Pretrain Validation Accuracy: Mean = 0.472, Std = 0.104
GRPO Validation Accuracy: Mean = 0.528, Std = 0.142

## Cross-species 

Pretrain Validation Accuracy: Mean = 0.798, Std = 0.056
GRPO Validation Accuracy: Mean = 0.817, Std = 0.049

## Oil 

Pretrain Validation Accuracy: Mean = 0.325, Std = 0.068
GRPO Validation Accuracy: Mean = 0.365, Std = 0.045

References:
1. Vaswani, A. (2017). 
    Attention is all you need. 
    Advances in Neural Information Processing Systems.
2. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., ... & He, Y. (2025). 
    DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. 
    arXiv preprint arXiv:2501.12948.
3. Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Hassabis, D. (2017). 
    Mastering the game of go without human knowledge. 
    nature, 550(7676), 354-359.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from util import create_data_module
import numpy as np

class TransformerClassifier(nn.Module):
    def __init__(
        self, 
        nfeatures: int = 2080, 
        ninp: int = 256, 
        nhead: int = 8, 
        nhid: int = 512, 
        nlayers: int = 3, 
        nclasses: int = 2, 
        dropout: int = 0.5
    ) -> None:
        """ 
        Transformer-based classifier.

        Args:
            nfeatures (int): The number of input features.
            ninp (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            nhid (int): The dimension of the feedforward network model.
            nlayers (int): The number of sub-encoder-layers in the encoder.
            nclasses (int): The number of classes in the dataset.
            dropout (float): The dropout value. Defaults to 0.5.
        """
        super(TransformerClassifier, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(nfeatures, ninp * 2),
            nn.LayerNorm(ninp * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ninp * 2, ninp),
            nn.LayerNorm(ninp),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        
        self.decoder = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.LayerNorm(nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nclasses)
        )
        
        self.ninp = ninp

    def forward(
        self, 
        src: torch.Tensor
    ) -> torch.Tensor:
        """ 
        Forward pass through the model.

        Args: 
            src (Tensor): The input features.

        Returns: 
            output (Tensor): The output logits.
        """
        src = self.encoder(src)
        src = src.unsqueeze(1)
        output = self.transformer(src)
        output = self.decoder(output.squeeze(1))
        return output

def compute_rewards(
        model: nn.Module, 
        src: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
    """ 
    Compute rewards for the given model and data.

    Args:
        model (nn.Module): The model to evaluate.
        src (Tensor): The input features.
        labels (Tensor): The target labels.

    Returns: 
        rewards (Tensor): The rewards for the current policy.
        accuracy (float): The accuracy of the model.
    """
    with torch.no_grad():
        logits = model(src)  # [batch_size, num_classes]
        probs = F.softmax(logits, dim=-1)
        labels = torch.argmax(labels, dim=-1)
        
        # Reward = 1.0 if correct, 0.0 otherwise
        preds = torch.argmax(probs, dim=1)
        rewards = (preds == labels).float()
        
        # Add a small bonus for high-confidence correct predictions
        correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()
        rewards = torch.clamp(rewards + correct_probs * 0.1, 0.0, 1.0)  # Ensure rewards are in [0, 1]
        
        # Calculate accuracy separately
        accuracy = (preds == labels).float().mean()
        
    return rewards, accuracy

def grpo_loss(
        policy: torch.Tensor, 
        old_policy: torch.Tensor, 
        rewards: torch.Tensor, 
        target_kl: float = 0.01
    ) -> torch.Tensor:
    """
    Compute the Generalized Reward-Per-Optimization (GRPO) loss.

    Args: 
        policy (Tensor): The current policy.
        old_policy (Tensor): The policy from the previous iteration.
        rewards (Tensor): The rewards for the current policy.
        target_kl (float): The target KL divergence between old and new policies. Defaults to 0.01.

    Returns:
        loss (Tensor): The GRPO loss.
    """
    log_probs = F.log_softmax(policy, dim=1)  # [batch_size, num_classes]
    old_log_probs = F.log_softmax(old_policy, dim=1)
    ratios = torch.exp(log_probs - old_log_probs)
    
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # [batch_size]
    advantages = advantages.unsqueeze(1)  # [batch_size, 1]
    
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
    
    policy_loss = -torch.min(surr1, surr2).mean()
    kl_loss = F.kl_div(log_probs, old_log_probs.exp(), reduction='batchmean')
    
    return policy_loss + target_kl * kl_loss

def train_grpo(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    src: torch.Tensor, 
    old_policy: torch.Tensor, 
    rewards: torch.Tensor, 
    max_kl: float = 0.1
) -> (float, torch.Tensor):
    """ 
    Perform a single update step using the Generalized Reward-Per-Optimization (GRPO) algorithm.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        src (Tensor): The input features.
        old_policy (Tensor): The policy from the previous iteration.
        rewards (Tensor): The rewards for the current policy.
        max_kl (float): The maximum KL divergence between old and new policies. Defaults to 0.1.

    Returns: 
        loss (float): The loss value.
        policy (Tensor): The new policy.
    """
    loss = torch.tensor(0.0)
    policy = None
    
    for i in range(3):  # Perform 3 updates per batch
        model.train()
        optimizer.zero_grad()
        
        policy = model(src)  # [batch_size, num_classes]
        
        loss = grpo_loss(policy, old_policy, rewards)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
    
    return loss.item(), policy

def pretrain(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    epochs: int = 50
) -> nn.Module:
    """
    Pretrain the model using cross-entropy loss. 

    Args: 
        model (nn.Module): The model to train.
        train_loader (DataLoader): The training DataLoader.
        val_loader (DataLoader): The validation DataLoader.
        epochs (int): The number of epochs to train for.

    Returns: 
        model (nn.Module): The trained model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5)
    
    # Early stopping parameters
    best_val_acc = 0.0
    patience = 50  # Number of epochs to wait for improvement before stopping
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_acc = []
        for src, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(src)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == torch.argmax(labels, dim=1)).float().mean()
            train_acc.append(acc.item())
        
        model.eval()
        val_acc = []
        with torch.no_grad():
            for src, labels in val_loader:
                outputs = model(src)
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == torch.argmax(labels, dim=1)).float().mean()
                val_acc.append(acc.item())
        
        val_acc_mean = sum(val_acc)/len(val_acc)
        scheduler.step(val_acc_mean)
        
        print(f'Epoch {epoch}:')
        print(f'  Train Accuracy: {sum(train_acc)/len(train_acc):.3f}')
        print(f'  Val Accuracy: {val_acc_mean:.3f}')
        
        # Early stopping logic
        if val_acc_mean > best_val_acc:
            best_val_acc = val_acc_mean
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch} (best Val Accuracy: {best_val_acc:.3f})')
            break
    
    return model

def load_data(
    dataset: str = "species"
) -> (TensorDataset, list):
    """ 
    Load the specified dataset and scale the features using Standard.

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
    dataset = "species"
    # Load data
    scaled_dataset, targets = load_data(dataset=dataset)
    
    # Initialize lists to store results
    pretrain_results = []
    grpo_results = []
    
    # Perform Stratified Cross-Validation
    n_splits = 3 if dataset == "part" else 5 # Not enough classes for 5-fold cross-validation on the "part" dataset.
    n_classes = {"species": 2, "part": 7, "oil": 7, "cross-species": 3}
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_dataset, targets)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # Create train/val datasets for this fold
        train_dataset = Subset(scaled_dataset, train_idx)
        val_dataset = Subset(scaled_dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = TransformerClassifier(
            nfeatures=2080, 
            ninp=256, 
            nhead=8, 
            nhid=512, 
            nlayers=3, 
            nclasses=n_classes[dataset], 
            dropout=0.3
        )
        
        # Step 1: Pretrain with cross-entropy
        print("Pretraining with cross-entropy...")
        model = pretrain(model, train_loader, val_loader, epochs=50)
        
        # Evaluate pretrained model
        model.eval()
        val_accuracies = []
        with torch.no_grad():
            for src, labels in val_loader:
                rewards, acc = compute_rewards(model, src, labels)
                val_accuracies.append(acc.item())
        
        pretrain_val_acc = sum(val_accuracies)/len(val_accuracies)
        pretrain_results.append(pretrain_val_acc)
        print(f'Pretrain Validation Accuracy: {pretrain_val_acc:.3f}')
        
        # Step 2: Fine-tune with GRPO
        print("\nFine-tuning with GRPO...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5)
        
        # Early stopping parameters
        best_val_acc = 0.0
        patience = 50 # Perform full training, even if early stopping is triggered
        patience_counter = 0
        
        for epoch in range(50):
            train_accuracies = []
            for src, labels in train_loader:
                with torch.no_grad():
                    old_policy = model(src)
                rewards, acc = compute_rewards(model, src, labels)
                train_accuracies.append(acc.item())
                loss, new_policy = train_grpo(model, optimizer, src, old_policy, rewards)
            
            val_accuracies = []
            model.eval()
            with torch.no_grad():
                for src, labels in val_loader:
                    rewards, acc = compute_rewards(model, src, labels)
                    val_accuracies.append(acc.item())
            
            val_acc = sum(val_accuracies)/len(val_accuracies)
            scheduler.step(val_acc)
            
            print(f'Epoch {epoch}:')
            print(f'  Train Accuracy: {sum(train_accuracies)/len(train_accuracies):.3f}')
            print(f'  Val Accuracy: {val_acc:.3f}')
            
            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} (best Val Accuracy: {best_val_acc:.3f})')
                break
        
        grpo_results.append(best_val_acc)
        print(f'GRPO Validation Accuracy: {best_val_acc:.3f}')
    
    # Report mean and standard deviation of results
    pretrain_mean = np.mean(pretrain_results)
    pretrain_std = np.std(pretrain_results)
    grpo_mean = np.mean(grpo_results)
    grpo_std = np.std(grpo_results)
    
    print("\nFinal Results:")
    print(f'Pretrain Validation Accuracy: Mean = {pretrain_mean:.3f}, Std = {pretrain_std:.3f}')
    print(f'GRPO Validation Accuracy: Mean = {grpo_mean:.3f}, Std = {grpo_std:.3f}')

if __name__ == "__main__":
    main()
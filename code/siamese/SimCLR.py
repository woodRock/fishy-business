import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

def setup_logger(name: str) -> logging.Logger:
    """Set up logger with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "contrastive_training.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Create logger instance
logger = setup_logger(__name__)

class EncoderNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], embedding_dim=128):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)  # Increased dropout
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Simplified projection head to reduce overfitting
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], embedding_dim),
            nn.BatchNorm1d(embedding_dim)  # Added batch norm
        )

    def forward(self, x):
        x = x.float()
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, pos_weight=0.985, neg_weight=0.015):
        super().__init__()
        self.temperature = temperature
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    
    def forward(self, z1, z2, labels):
        batch_size = z1.shape[0]
        
        # Safe normalization
        z1_norm = torch.norm(z1, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        z2_norm = torch.norm(z2, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        z1 = z1 / z1_norm
        z2 = z2 / z2_norm
        
        # Get class indices
        label_indices = torch.argmax(labels, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(z1, z2.T) / self.temperature
        similarity = similarity.clamp(min=-1e2, max=1e2)
        
        # Create mask for similar pairs (same class)
        pos_mask = (label_indices.unsqueeze(0) == label_indices.unsqueeze(1)).float()
        neg_mask = 1 - pos_mask
        
        # Apply inverse class weights
        pos_weight = torch.where(label_indices == 1, 
                               torch.tensor(self.neg_weight, device=z1.device),
                               torch.tensor(self.pos_weight, device=z1.device))
        
        # Weight the positive and negative contributions
        weighted_pos = pos_mask * pos_weight.unsqueeze(1)
        weighted_neg = neg_mask * pos_weight.unsqueeze(1)
        
        # Compute weighted contrastive loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        pos_loss = -weighted_pos * log_prob
        neg_loss = -weighted_neg * log_prob
        
        # Combine losses with class balancing
        loss = (pos_loss.sum() + neg_loss.sum()) / (weighted_pos.sum() + weighted_neg.sum() + 1e-8)
        
        return loss
    
def compute_accuracy(embeddings_1, embeddings_2, labels):
    """
    Compute balanced classification accuracy based on embedding similarity.
    """
    # Convert one-hot labels to class indices
    labels = torch.argmax(labels, dim=1).cpu().numpy()
    
    # Normalize embeddings
    z1 = F.normalize(embeddings_1, dim=1)
    z2 = F.normalize(embeddings_2, dim=1)
    similarity = F.cosine_similarity(z1, z2).cpu().numpy()
    
    # Debug embeddings and similarities
    print("\nEmbedding Statistics:")
    print(f"Embedding 1 mean: {embeddings_1.mean().item():.3f}, std: {embeddings_1.std().item():.3f}")
    print(f"Embedding 2 mean: {embeddings_2.mean().item():.3f}, std: {embeddings_2.std().item():.3f}")
    
    # Get unique labels and their counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution: {list(zip(unique_labels, label_counts))}")
    
    # Create true labels for pairs (same or different class)
    true_labels = labels.astype(int)
    
    # Analyze similarity distribution
    sim_mean = similarity.mean()
    sim_std = similarity.std()
    print("\nSimilarity Statistics:")
    print(f"Overall similarity: mean={sim_mean:.3f}, std={sim_std:.3f}")
    print(f"Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
    
    # Use adaptive threshold
    threshold = sim_mean
    predictions = (similarity > threshold).astype(int)
    
    print(f"\nPrediction Statistics:")
    print(f"Threshold: {threshold:.3f}")
    pred_unique, pred_counts = np.unique(predictions, return_counts=True)
    print(f"Prediction distribution: {list(zip(pred_unique, pred_counts))}")
    
    true_unique, true_counts = np.unique(true_labels, return_counts=True)
    print(f"True label distribution: {list(zip(true_unique, true_counts))}")
    
    return balanced_accuracy_score(true_labels, predictions)

def evaluate_model(model, loader, device):
    """
    Evaluate model on a data loader.
    Returns accuracy and loss.
    """
    model.eval()
    total_loss = 0
    all_embeddings_1 = []
    all_embeddings_2 = []
    all_labels = []
    
    criterion = SupervisedContrastiveLoss()
    
    with torch.no_grad():
        for spec1, spec2, labels in loader:
            spec1 = spec1.float().to(device)
            spec2 = spec2.float().to(device)
            labels = labels.float().to(device)
            
            # Get embeddings
            z1 = model(spec1)
            z2 = model(spec2)
            
            # Store for accuracy computation
            all_embeddings_1.append(z1)
            all_embeddings_2.append(z2)
            all_labels.append(labels)
            
            # Compute loss
            loss = criterion(z1, z2, labels)
            total_loss += loss.item()
    
    # Concatenate all batches
    embeddings_1 = torch.cat(all_embeddings_1)
    embeddings_2 = torch.cat(all_embeddings_2)
    labels = torch.cat(all_labels)
        
    # Compute metrics
    accuracy = compute_accuracy(embeddings_1, embeddings_2, labels)
    avg_loss = total_loss / len(loader)
    
    return accuracy, avg_loss

def train_contrastive_model(train_loader: DataLoader,
                          val_loader: DataLoader,
                          input_dim: int,
                          device: str = 'cuda',
                          hidden_dims: list = [512, 256, 128],  # Smaller network
                          embedding_dim: int = 64,  # Smaller embedding
                          temperature: float = 0.1,  
                          epochs: int = 100,
                          learning_rate: float = 1e-4) -> EncoderNetwork:
    """
    Train contrastive learning model using provided data loaders.
    """
    logger.info(f"Starting training with:")
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Hidden dimensions: {hidden_dims}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Device: {device}")
    
    # Initialize model, loss, and optimizer
    model = EncoderNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim
    ).to(device)
    
    criterion = SupervisedContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (spec1, spec2, labels) in enumerate(train_loader):
            spec1 = spec1.float().to(device)
            spec2 = spec2.float().to(device)
            labels = labels.float().to(device)
            
            # Forward pass
            z1 = model(spec1)
            z2 = model(spec2)
            
            # Compute loss
            loss = criterion(z1, z2, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] - Batch [{batch_idx+1}/{len(train_loader)}]: "
                          f"Loss = {loss.item():.4f}")
        
        # Evaluate on train set
        train_acc, avg_train_loss = evaluate_model(model, train_loader, device)
        
        # Evaluate on validation set
        val_acc, avg_val_loss = evaluate_model(model, val_loader, device)
        
        # Log progress every epoch
        logger.info(
            f"Epoch [{epoch+1}/{epochs}]: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Train Acc = {train_acc:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, "
            f"Val Acc = {val_acc:.4f}"
        )
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")

        # Update learning rate
        scheduler.step()
    
    # Load best model
    model.load_state_dict(best_state)
    logger.info(f"Training completed! Final best validation accuracy: {best_val_acc:.4f}")
    return model

def get_embeddings(model: EncoderNetwork, 
                  loader: DataLoader,
                  device: str = 'cuda') -> torch.Tensor:
    """
    Get embeddings for all samples in the loader.
    
    Args:
        model: Trained encoder network
        loader: DataLoader containing samples
        device: Device to compute embeddings on
        
    Returns:
        Tensor of embeddings
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for spec1, spec2, _ in loader:
            spec1 = spec1.float().to(device)
            spec2 = spec2.float().to(device)
            
            # Get embeddings for both spectra
            z1 = model(spec1)
            z2 = model(spec2)
            
            # Store embeddings
            embeddings.extend([z1.cpu(), z2.cpu()])
    
    return torch.cat(embeddings, dim=0)

if __name__ == "__main__":
    # Example usage with your SiameseDataset
    from util import DataConfig, prepare_dataset
    
    logger.info("Starting main program")
    
    # Get data loaders
    config = DataConfig()
    logger.info("Preparing datasets...")
    train_loader, val_loader = prepare_dataset(config)
    
    # Get input dimension from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[1]
    logger.info(f"Data loaded successfully. Input dimension: {input_dim}")
    
    # Train model
    logger.info("Starting model training...")
    model = train_contrastive_model(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Final evaluation
    train_acc, train_loss = evaluate_model(model, train_loader, 'cuda' if torch.cuda.is_available() else 'cpu')
    val_acc, val_loss = evaluate_model(model, val_loader, 'cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Final Results:")
    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Train Loss: {train_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Validation Loss: {val_loss:.4f}")

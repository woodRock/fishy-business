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
    """Encoder network for mass spectrometry data"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], embedding_dim=64):
        super().__init__()
        
        # Encoder layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim, dtype=torch.float32),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*layers)
        self.embedding = nn.Linear(hidden_dims[-1], embedding_dim, dtype=torch.float32)

    def forward(self, x):
        x = x.float()
        h = self.encoder(x)
        z = self.embedding(h)
        return F.normalize(z, dim=1)

class SupervisedContrastiveLoss(nn.Module):
    """Contrastive loss for one-hot encoded labels"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2, labels):
        """
        Args:
            z1: First set of embeddings (batch_size, embedding_dim)
            z2: Second set of embeddings (batch_size, embedding_dim)
            labels: One-hot encoded labels (batch_size, num_classes)
        """
        # Concatenate embeddings from both views
        embeddings = torch.cat([z1, z2], dim=0)  # (2*batch_size, embedding_dim)
        
        # Duplicate labels for both views
        labels_duplicated = torch.cat([labels, labels], dim=0)  # (2*batch_size, num_classes)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (2*batch_size, 2*batch_size)
        
        # Scale similarities by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Compute label similarity matrix (indicates which pairs have the same label)
        label_similarity = torch.matmul(labels_duplicated, labels_duplicated.T)  # (2*batch_size, 2*batch_size)
        
        # Remove self-similarities
        mask_self = ~torch.eye(similarity_matrix.shape[0], dtype=bool, device=embeddings.device)
        
        # Apply mask to remove self-similarities
        similarity_matrix = similarity_matrix[mask_self].view(similarity_matrix.shape[0], -1)
        label_similarity = label_similarity[mask_self].view(label_similarity.shape[0], -1)
        
        # Compute log probabilities
        log_prob = similarity_matrix - torch.logsumexp(similarity_matrix, dim=1, keepdim=True)
        
        # Compute mean of positive similarities
        mean_log_prob_pos = (label_similarity * log_prob).sum(1) / label_similarity.sum(1).clamp(min=1e-8)
        
        return -mean_log_prob_pos.mean()

def compute_accuracy(embeddings_1, embeddings_2, labels):
    """
    Compute balanced classification accuracy based on embedding similarity.
    Args:
        embeddings_1: First set of embeddings (n_samples, embedding_dim)
        embeddings_2: Second set of embeddings (n_samples, embedding_dim)
        labels: One-hot encoded labels (n_samples, n_classes)
    Returns:
        float: Balanced classification accuracy
    """
    # Convert one-hot labels to class indices
    labels = torch.argmax(labels, dim=1).cpu().numpy()
    
    # Compute cosine similarity between pairs
    similarity = F.cosine_similarity(embeddings_1, embeddings_2)
    
    # Convert similarities to binary predictions (1 if similar, 0 if different)
    predictions = (similarity > 0).cpu().numpy()
    
    # Convert to same/different class labels
    true_labels = (labels == labels).astype(int)
    
    # Ensure predictions and true_labels have the same shape
    assert len(predictions) == len(true_labels), f"Predictions shape {predictions.shape} != True labels shape {true_labels.shape}"
    
    # Compute balanced accuracy
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
                          hidden_dims: list = [512, 256, 128],
                          embedding_dim: int = 64,
                          temperature: float = 0.5,
                          epochs: int = 100,
                          learning_rate: float = 1e-3) -> EncoderNetwork:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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

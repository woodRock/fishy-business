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

class EncoderBlock(nn.Module):
    """
    Encoder block with LayerNorm, residual connections, and dropout.
    """
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual connection if dimensions match
        self.has_residual = in_dim == out_dim
        if not self.has_residual:
            self.residual_proj = nn.Linear(in_dim, out_dim)
            
    def forward(self, x):
        if self.has_residual:
            return self.layer(x) + x
        else:
            return self.layer(x) + self.residual_proj(x)


class EncoderNetwork(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dims=[4096, 2048, 1024, 512, 256], 
        embedding_dim=512,
        dropout_rate=0.1
    ):
        super().__init__()
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Encoder layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(EncoderBlock(prev_dim, dim, dropout_rate))
            prev_dim = dim
            
        self.encoder = nn.Sequential(*layers)
        
        # Projection head with multiple layers
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x.float())
        
        # Encoder
        h = self.encoder(x)
        
        # Projection
        z = self.projection(h)
        
        # Normalize embeddings to unit sphere
        return F.normalize(z, dim=1)

class ContrastiveLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, z1, z2, labels):
        z1 = F.normalize(z1 + 1e-8, dim=1)
        z2 = F.normalize(z2 + 1e-8, dim=1)
        
        similarities = F.cosine_similarity(z1, z2)
        label_pairs = torch.argmax(labels, dim=1)
        
        # Scale similarities more aggressively
        probs = torch.clamp((similarities + 1) / 2, min=1e-6, max=1-1e-6)
        
        # Remove label smoothing to allow perfect classification
        loss = F.binary_cross_entropy(probs, label_pairs.float())
        
        return loss

def compute_accuracy(embeddings_1, embeddings_2, labels):
    """Basic accuracy computation"""
    with torch.no_grad():  # Ensure no gradients are computed
        z1 = F.normalize(embeddings_1, dim=1)
        z2 = F.normalize(embeddings_2, dim=1)
        labels = torch.argmax(labels, dim=1).cpu().numpy()
        
        # Use cosine similarity and detach tensors
        similarity = F.cosine_similarity(z1.detach(), z2.detach()).cpu().numpy()
        
        # Simple threshold at mean
        threshold = np.mean(similarity)
        predictions = (similarity > threshold).astype(int)
        
        return balanced_accuracy_score(labels, predictions)

def train_contrastive_model(train_loader, val_loader, input_dim, device='cuda',
                          hidden_dims=[1024, 512, 256, 128], embedding_dim=128, epochs=1000):
    
    model = EncoderNetwork(input_dim, hidden_dims, embedding_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        train_losses = []
        train_accuracies = []
        
        for spec1, spec2, labels in train_loader:
            spec1, spec2 = spec1.to(device), spec2.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            z1 = model(spec1)
            z2 = model(spec2)
            
            loss = criterion(z1, z2, labels)
            loss.backward()

            # Gradient clipping.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            train_acc = compute_accuracy(z1, z2, labels)
                
            train_losses.append(loss.item())
            train_accuracies.append(train_acc)
            
            train_loss += loss.item()

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accuracies)
        
        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for spec1, spec2, labels in val_loader:
                spec1, spec2 = spec1.to(device), spec2.to(device)
                labels = labels.to(device)
                
                z1 = model(spec1)
                z2 = model(spec2)
                
                val_loss = criterion(z1, z2, labels)
                val_acc = compute_accuracy(z1, z2, labels)
                
                val_losses.append(val_loss.item())
                val_accuracies.append(val_acc)
        
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accuracies)
        
        print(f"Epoch [{epoch+1}/{epochs}]: "
              f"Train Loss = {train_loss/len(train_loader):.4f}, "
              f"Train Acc = {avg_train_acc:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, "
              f"Val Acc = {avg_val_acc:.4f}")
        
        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_state = model.state_dict()
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

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
    
    criterion = ContrastiveLoss()
    
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
    logger.info(f"Validation Accuracy: {train_acc:.4f}")
    logger.info(f"Validation Loss: {val_loss:.4f}")
"""SimCLR implementation for contrastive learning.
This module defines the encoder network, contrastive loss, training and evaluation functions.
It uses a Siamese architecture to learn embeddings from pairs of spectra."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score


def setup_logger(name: str) -> logging.Logger:
    """Set up logger with both file and console handlers.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "contrastive_training.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# Create logger instance
logger = setup_logger(__name__)


class EncoderBlock(nn.Module):
    """
    Encoder block with LayerNorm, residual connections, and dropout.
    """

    def __init__(self, in_dim, out_dim, dropout_rate=0.1) -> None:
        """Initialize the encoder block.

        Args:
            in_dim (int): Input dimension of the block.
            out_dim (int): Output dimension of the block.
            dropout_rate (float): Dropout rate to apply after ReLU activation.
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Residual connection if dimensions match
        self.has_residual = in_dim == out_dim
        if not self.has_residual:
            self.residual_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """Forward pass through the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """
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
        dropout_rate=0.1,
    ) -> None:
        """Initialize the encoder network.

        This network consists of multiple encoder blocks followed by a projection head.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (list): List of hidden layer dimensions.
            embedding_dim (int): Dimension of the final embedding.
            dropout_rate (float): Dropout rate to apply after ReLU activation.
        """
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
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x):
        """Forward pass through the encoder network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Normalized embeddings of shape (batch_size, embedding_dim).
        """
        # Input normalization
        x = self.input_norm(x.float())

        # Encoder
        h = self.encoder(x)

        # Projection
        z = self.projection(h)

        # Normalize embeddings to unit sphere
        return F.normalize(z, dim=1)


class ContrastiveLoss(nn.Module):
    """Contrastive loss based on InfoNCE."""

    def __init__(self, temperature=0.07) -> None:
        """Initialize the contrastive loss with temperature scaling.

        Args:
            temperature (float): Temperature parameter for scaling the similarity.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, labels):
        """Forward pass to compute the contrastive loss.

        A custom contrastive loss function that computes the NT-Xent Loss (InfoNCE) for two sets of embeddings.

        Args:
            z1 (torch.Tensor): Embeddings from the first view of the batch.
            z2 (torch.Tensor): Embeddings from the second view of the batch.
            labels (torch.Tensor): Labels for the pairs, these are not used in the loss computation but can be useful for debugging.
        Returns:
            torch.Tensor: Computed contrastive loss.
        """

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute similarity matrix
        batch_size = z1.shape[0]
        features = torch.cat([z1, z2], dim=0)
        similarity = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=2
        )

        # Apply temperature scaling
        similarity = similarity / self.temperature

        # Create mask for positive pairs
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=z1.device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)

        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -(pos_mask * log_prob).sum() / (2 * batch_size)

        return loss


def compute_accuracy(embeddings_1, embeddings_2, labels):
    """Basic accuracy computation.

    Args:
        embeddings_1 (torch.Tensor): First set of embeddings.
        embeddings_2 (torch.Tensor): Second set of embeddings.
        labels (torch.Tensor): True labels for the pairs.

    Returns:
        float: Balanced accuracy score for the predictions.
    """
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


def train_contrastive_model(
    train_loader,
    val_loader,
    input_dim,
    device="cuda",
    hidden_dims=[1024, 512, 256, 128],
    embedding_dim=128,
    epochs=1000,
) -> nn.Module:
    """Train a contrastive model using the provided data loaders.

    This function trains a Siamese encoder network using contrastive loss.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        input_dim (int): Dimension of the input features.
        device (str): Device to run the model on ("cuda" or "cpu").
        hidden_dims (list): List of hidden layer dimensions for the encoder.
        embedding_dim (int): Dimension of the final embedding.
        epochs (int): Number of training epochs.

    Returns:
        nn.Module: Trained encoder network.
    """
    model = EncoderNetwork(input_dim, hidden_dims, embedding_dim).to(device)
    criterion = ContrastiveLoss(temperature=0.07)

    # Use larger batch size if possible
    # Add weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    # Cosine learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    best_state = None
    patience = 1000
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_accuracies = []

        for spec1, spec2, labels in train_loader:
            spec1, spec2 = spec1.to(device), spec2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Add gradient accumulation for larger effective batch size
            z1 = model(spec1)
            z2 = model(spec2)

            loss = criterion(z1, z2, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_acc = compute_accuracy(z1, z2, labels)
            train_losses.append(loss.item())
            train_accuracies.append(train_acc)

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

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accuracies)
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accuracies)

        # Early stopping with patience
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(
            f"Epoch [{epoch+1}/{epochs}]: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Train Acc = {avg_train_acc:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, "
            f"Val Acc = {avg_val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_model(model, loader, device):
    """
    Evaluate model on a data loader.
    Returns accuracy and loss.

    This function computes the embeddings for pairs of spectra, calculates the contrastive loss,
    and evaluates the accuracy of the model on the dataset.

    Args:
        model: Trained encoder network
        loader: DataLoader containing samples
        device: Device to compute embeddings on

    Returns:
        Tuple[float, float]: Accuracy and average loss on the dataset.
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


def get_embeddings(
    model: EncoderNetwork, loader: DataLoader, device: str = "cuda"
) -> torch.Tensor:
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
    """Entry point for the script.
    Parses command line arguments, initializes the configuration, and calls the main function.
    """
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
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Final evaluation
    train_acc, train_loss = evaluate_model(
        model, train_loader, "cuda" if torch.cuda.is_available() else "cpu"
    )
    val_acc, val_loss = evaluate_model(
        model, val_loader, "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Final Results:")
    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Train Loss: {train_loss:.4f}")
    logger.info(f"Validation Accuracy: {train_acc:.4f}")
    logger.info(f"Validation Loss: {val_loss:.4f}")

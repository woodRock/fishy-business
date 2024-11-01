from dataclasses import dataclass
import logging
import random
from typing import List, Tuple, Union, Optional, TypeVar

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer import Transformer

# Type aliases
T = TypeVar("T", bound=torch.Tensor)
TensorPair = Tuple[torch.Tensor, torch.Tensor]
Device = Union[str, torch.device]


@dataclass
class PreTrainingConfig:
    """Configuration for pre-training tasks."""

    num_epochs: int = 100
    file_path: str = "transformer_checkpoint.pth"
    n_features: int = 2080
    chunk_size: int = 50
    device: Device = "cuda" if torch.cuda.is_available() else "cpu"


def mask_spectra_side(input_spectra: T, side: str = "left") -> T:
    """Mask either left or right side of the input spectra.

    Args:
        input_spectra: Input spectra tensor of shape (batch_size, n_features)
        side: Which side to mask ('left' or 'right')

    Returns:
        Masked spectra tensor

    Raises:
        ValueError: If side is not 'left' or 'right'
    """
    if side not in ["left", "right"]:
        raise ValueError("side must be either 'left' or 'right'")

    split_index = input_spectra.shape[0] // 2
    masked_spectra = input_spectra.clone()

    if side == "left":
        masked_spectra[:split_index] = 0
    else:
        masked_spectra[split_index:] = 0

    return masked_spectra


class PreTrainer:
    """Handles pre-training tasks for transformer models."""

    def __init__(
        self,
        model: Transformer,
        config: PreTrainingConfig,
        criterion: Optional[CrossEntropyLoss] = None,
        optimizer: Optional[AdamW] = None,
    ):
        """Initialize pre-trainer with model and configuration.

        Args:
            model: Transformer model to pre-train
            config: Pre-training configuration
            criterion: Loss function (defaults to CrossEntropyLoss)
            optimizer: Optimizer (defaults to AdamW)
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

        if criterion is None:
            criterion = CrossEntropyLoss()
        if optimizer is None:
            optimizer = AdamW(model.parameters())

        self.criterion = criterion
        self.optimizer = optimizer

    def pre_train_masked_spectra(self, train_loader: DataLoader) -> Transformer:
        """Pre-train using masked spectra modeling with progressive masking.

        Args:
            train_loader: DataLoader containing training data

        Returns:
            Pre-trained transformer model
        """
        for epoch in tqdm(
            range(self.config.num_epochs), desc="Pre-training: Masked Spectra"
        ):
            total_loss = 0.0
            self.model.train()

            for x, _ in train_loader:
                x = x.to(self.config.device)
                batch_size = x.shape[0]

                # Process features in chunks to manage memory
                for start_idx in range(
                    1, self.config.n_features, self.config.chunk_size
                ):
                    end_idx = min(
                        start_idx + self.config.chunk_size, self.config.n_features
                    )

                    # Create and apply mask
                    mask = torch.zeros(
                        batch_size, self.config.n_features, dtype=torch.bool
                    )
                    mask = mask.to(self.config.device)
                    mask[:, start_idx:end_idx] = True

                    masked_x = x.clone()
                    masked_x[mask] = 0

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(masked_x, masked_x)
                    target = x[:, start_idx:end_idx]

                    # Calculate loss and update
                    loss = self.criterion(outputs[:, start_idx:end_idx], target)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

            avg_loss = total_loss / (len(train_loader) * (self.config.n_features - 1))
            self.logger.info(
                f"Epoch [{epoch+1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}"
            )

        torch.save(self.model.state_dict(), self.config.file_path)
        return self.model

    def pre_train_next_spectra(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Transformer:
        """Pre-train using Next Spectra Prediction (NSP).

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Pre-trained transformer model
        """
        # Generate contrastive pairs
        train_pairs = self._generate_contrastive_pairs(train_loader)
        val_pairs = self._generate_contrastive_pairs(val_loader)

        for epoch in tqdm(
            range(self.config.num_epochs), desc="Pre-training: Next Spectra"
        ):
            # Training phase
            train_loss = self._run_epoch(train_pairs, training=True)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_epoch(val_pairs, training=False)

            self.logger.info(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        torch.save(self.model.state_dict(), self.config.file_path)
        return self.model

    def _generate_contrastive_pairs(
        self, data_loader: DataLoader
    ) -> List[Tuple[TensorPair, List[float]]]:
        """Generate contrastive pairs for NSP training.

        Args:
            data_loader: Input data loader

        Returns:
            List of (input_pair, label) tuples
        """
        pairs = []

        for x, _ in data_loader:
            for i in range(len(x)):
                if random.random() < 0.5 and i < len(x) - 1:
                    # Same sequence pairs
                    left = mask_spectra_side(x[i], "right")
                    right = mask_spectra_side(x[i], "left")
                    pairs.append(((left, right), [1, 0]))
                else:
                    # Different sequence pairs
                    j = random.choice([k for k in range(len(x)) if k != i])
                    left = mask_spectra_side(x[i], "right")
                    right = mask_spectra_side(x[j], "left")
                    pairs.append(((left, right), [0, 1]))

        return pairs

    def _run_epoch(
        self, pairs: List[Tuple[TensorPair, List[float]]], training: bool = True
    ) -> float:
        """Run a single epoch of training or validation.

        Args:
            pairs: List of (input_pair, label) tuples
            training: Whether this is a training epoch

        Returns:
            Average loss for the epoch
        """
        total_loss = 0.0

        for (left, right), label in pairs:
            left = left.to(self.config.device)
            right = right.to(self.config.device)
            label = torch.tensor(label, dtype=torch.float).to(self.config.device)

            if training:
                self.optimizer.zero_grad()

            output = self.model(left.unsqueeze(0), right.unsqueeze(0))
            loss = self.criterion(output, label.unsqueeze(0))

            if training:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(pairs)


def load_pretrained_weights(
    model: Transformer,
    file_path: str = "transformer_checkpoint.pth",
    output_dim: int = 2,
) -> Transformer:
    """Load pre-trained weights and adjust output dimension.

    Args:
        model: Target transformer model
        file_path: Path to checkpoint file
        output_dim: Number of output classes

    Returns:
        Model with loaded pre-trained weights
    """
    checkpoint = torch.load(file_path)

    # Adjust final layer dimensions
    checkpoint["fc.weight"] = checkpoint["fc.weight"][:output_dim]
    checkpoint["fc.bias"] = checkpoint["fc.bias"][:output_dim]

    model.load_state_dict(checkpoint, strict=False)
    return model

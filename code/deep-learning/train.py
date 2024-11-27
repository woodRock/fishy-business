from tqdm import tqdm
import logging
import copy
import time
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    auc,
    roc_curve,
)
from transformer import Transformer
from vae import VAE
from util import DataAugmenter, AugmentationConfig
from plot import plot_accuracy, plot_confusion_matrix

MetricsDict = Dict[str, float]
FoldMetrics = Dict[str, List]

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 100,
    patience: int = 10,
    n_splits: int = 5,
    is_augmented: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """Train a model using k-fold cross-validation with early stopping.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader containing training data
        criterion: Loss function
        optimizer: Optimizer instance
        num_epochs: Maximum number of training epochs
        patience: Number of epochs to wait before early stopping
        n_splits: Number of folds for cross-validation
        is_augmented: Whether to apply data augmentation
        device: Device to train on ('cuda' or 'cpu')

    Returns:
        Trained model with best performance across all folds
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting k-fold cross validation training")

    model_copy = copy.deepcopy(model)
    dataset = train_loader.dataset

    # Extract labels for stratification.
    all_labels = _extract_labels(dataset)

    # Initialize cross-validation.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Metrics storage
    fold_metrics: FoldMetrics = {
        "train_losses": [],
        "val_losses": [],
        "train_metrics": [],
        "val_metrics": [],
    }
    best_val_metrics: List[MetricsDict] = []
    best_model_state = None
    best_overall_accuracy = float("-inf")

    # Perform k-fold cross-validation.
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(dataset)), all_labels), 1
    ):
        logger.info(f"\nStarting Fold {fold}/{n_splits}")

        # Setup fold-specific data and model.
        # Reset the model each fold.
        model = copy.deepcopy(model_copy).to(device)
        fold_train_loader, fold_val_loader = _create_fold_loaders(
            dataset, train_idx, val_idx, train_loader.batch_size
        )

        # Apply data augmentation if enabled.
        if is_augmented:
            aug_config = AugmentationConfig(
                enabled=True,
                num_augmentations= 5,
                noise_enabled=True,
                shift_enabled=False,
                scale_enabled=False,
                noise_level=0.1,
                shift_range=0.1,
                scale_range=0.1,
            )

            # Augment the training set only.
            train_data_augmenter = DataAugmenter(aug_config)
            fold_train_loader = train_data_augmenter.augment(fold_train_loader)

        # Create fold-specific optimizer.
        fold_optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)

        # Train the fold
        fold_results = _train_fold(
            model=model,
            train_loader=fold_train_loader,
            val_loader=fold_val_loader,
            criterion=criterion,
            optimizer=fold_optimizer,
            num_epochs=num_epochs,
            patience=patience,
            device=device,
            logger=logger,
        )

        # Update best model if this fold performed better.
        if fold_results["best_accuracy"] > best_overall_accuracy:
            best_overall_accuracy = fold_results["best_accuracy"]
            best_model_state = copy.deepcopy(fold_results["best_model_state"])
            logger.info(
                f"\nNew best overall accuracy: {best_overall_accuracy:.4f} (Fold {fold})"
            )

        # Store fold metrics.
        fold_metrics = _update_fold_metrics(fold_metrics, fold_results["epoch_metrics"])
        best_val_metrics.append(fold_results["best_fold_metrics"])

    # Print final metrics.
    _print_final_metrics(best_val_metrics, logger)
    logger.info(
        f"\nTraining completed. Best overall accuracy: {best_overall_accuracy:.4f}"
    )

    # Load the best model state.
    model = model_copy.to(device)
    model.load_state_dict(best_model_state)

    return model


def transfer_learning(
    dataset: str, model: Transformer, file_path: str = "transformer_checkpoint.pth"
) -> Transformer:
    """Apply transfer learning by loading pre-trained weights and adapting the final layer.

    Args:
        dataset: Target dataset name ('species', 'part', 'oil', 'oil_simple', or 'cross-species')
        model: Transformer model to transfer weights to
        file_path: Path to pre-trained model checkpoint

    Returns:
        Model with transferred weights and adapted output layer

    Raises:
        ValueError: If dataset name is invalid
    """
    output_dims = {
        "species": 2,
        "oil_simple": 2,
        "part": 7,
        "oil": 7,
        "cross-species": 3,
    }

    if dataset not in output_dims:
        raise ValueError(
            f"Invalid dataset specified: {dataset}. Must be one of {list(output_dims.keys())}"
        )

    checkpoint = torch.load(file_path)
    output_dim = output_dims[dataset]

    # Adjust final layer weights
    if dataset in ["species", "oil_simple"]:
        checkpoint["fc.weight"] = checkpoint["fc.weight"][:output_dim]
        checkpoint["fc.bias"] = checkpoint["fc.bias"][:output_dim]
    else:
        checkpoint["fc.weight"] = torch.zeros(
            output_dim, checkpoint["fc.weight"].shape[1]
        )
        checkpoint["fc.bias"] = torch.zeros(output_dim)

    model.load_state_dict(checkpoint, strict=False)
    return model


def _calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> MetricsDict:
    """Calculate multiple classification metrics.

    Args:
        y_true: Ground truth labels as a 1D array
        y_pred: Predicted labels as a 1D array
        y_prob: Probability predictions for ROC AUC as a 2D array of shape (n_samples, n_classes)

    Returns:
        Dictionary containing calculated metrics:
            - balanced_accuracy: Balanced accuracy score
            - precision: Weighted precision score
            - recall: Weighted recall score
            - f1: Weighted F1 score
            - auc_roc: Area under ROC curve (if y_prob provided)
    """
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))

    metrics: MetricsDict = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=unique_classes
        ),
        "recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=unique_classes
        ),
        "f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=unique_classes
        ),
    }

    if y_prob is not None:
        if y_prob.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            metrics["auc_roc"] = auc(fpr, tpr)
        else:  # Multiclass classification
            n_classes = y_prob.shape[1]
            y_true_onehot = np.eye(n_classes)[y_true]
            aucs = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
                aucs.append(auc(fpr, tpr))
            metrics["auc_roc"] = np.mean(aucs)

    return metrics


# Helper functions
def _extract_labels(dataset: Dataset) -> np.ndarray:
    """Extract labels from dataset for stratification."""
    all_labels = []
    for _, labels in dataset:
        if isinstance(labels, torch.Tensor):
            if labels.dim() > 1:
                all_labels.append(labels.argmax().item())
            else:
                all_labels.append(labels.argmax(dim=0))
        elif isinstance(labels, np.ndarray):
            if labels.ndim > 1:
                all_labels.append(np.argmax(labels))
            else:
                all_labels.append(np.argmax(labels))
        else:
            all_labels.append(labels)
            
    return np.array(all_labels)


def _create_fold_loaders(
    dataset: Dataset, train_idx: np.ndarray, val_idx: np.ndarray, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for a specific fold."""
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True),
        DataLoader(val_subset, batch_size=batch_size, shuffle=False),
    )


def _train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    patience: int,
    device: str,
    logger: logging.Logger,
) -> Dict:
    """Train a single fold and return results.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer instance
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to train on
        logger: Logger instance

    Returns:
        Dictionary containing best model state and metrics
    """
    best_val_accuracy = float("-inf")
    epochs_without_improvement = 0
    best_model_state = None
    best_fold_metrics = None

    epoch_metrics = {
        "train_losses": [],
        "val_losses": [],
        "train_metrics": [],
        "val_metrics": [],
    }

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training phase.
        model.train()
        train_results = _run_epoch(
            model, train_loader, criterion, optimizer, device, True
        )

        # Validation phase.
        model.eval()
        with torch.no_grad():
            val_results = _run_epoch(
                model, val_loader, criterion, optimizer, device, False
            )

        # Store metrics for current epoch.
        _update_epoch_metrics(epoch_metrics, train_results, val_results)

        current_val_accuracy = val_results["metrics"]["balanced_accuracy"]

        # Update best model if improved.
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            best_fold_metrics = copy.deepcopy(val_results["metrics"])
            epochs_without_improvement = 0

            # Log when we find a better model.
            logger.info(
                f"Epoch {epoch + 1}: New best validation accuracy: {best_val_accuracy:.4f}"
            )

            # Log all metrics if we achieve perfect accuracy.
            if current_val_accuracy >= 1.0:
                logger.info("Achieved perfect validation accuracy!")
                logger.info("Current metrics:")
                for metric, value in val_results["metrics"].items():
                    logger.info(f"{metric}: {value:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
                # DEBUG
                # break

        # Log progress every few epochs.
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}")
            logger.info(f"Train Loss: {train_results['loss']:.4f}")
            logger.info(f"Val Loss: {val_results['loss']:.4f}")
            logger.info(f"Current Val Accuracy: {current_val_accuracy:.4f}")
            logger.info(f"Best Val Accuracy: {best_val_accuracy:.4f}")

    # Ensure we return the best metrics we found.
    return {
        "best_accuracy": best_val_accuracy,
        "best_model_state": best_model_state,
        "best_fold_metrics": best_fold_metrics,
        "epoch_metrics": epoch_metrics,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: str,
    is_training: bool,
) -> Dict:
    """Run a single epoch of training or validation."""
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    desc = "Training" if is_training else "Validation"

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if is_training:
            optimizer.zero_grad()

        if isinstance(model, VAE):
            _, _, _, outputs = model(inputs)
        else:
            outputs = (
                model(inputs, inputs)
                if isinstance(model, Transformer)
                else model(inputs)
            )
        loss = criterion(outputs, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        _, actual = labels.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(actual.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = _calculate_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )

    return {
        "loss": avg_loss,
        "metrics": metrics,
        "predictions": {
            "labels": np.array(all_labels),
            "preds": np.array(all_preds),
            "probs": np.array(all_probs),
        },
    }


def _update_fold_metrics(
    fold_metrics: FoldMetrics, epoch_metrics: Dict[str, List]
) -> FoldMetrics:
    """Update the fold metrics with the metrics from the current epoch.

    Args:
        fold_metrics: Dictionary containing metrics for all folds
        epoch_metrics: Dictionary containing metrics for current epoch

    Returns:
        Updated fold metrics dictionary
    """
    fold_metrics["train_losses"].append(epoch_metrics["train_losses"])
    fold_metrics["val_losses"].append(epoch_metrics["val_losses"])
    fold_metrics["train_metrics"].append(epoch_metrics["train_metrics"])
    fold_metrics["val_metrics"].append(epoch_metrics["val_metrics"])
    return fold_metrics


def _update_epoch_metrics(
    epoch_metrics: Dict[str, List], train_results: Dict, val_results: Dict
) -> None:
    """Update the epoch metrics with results from training and validation.

    Args:
        epoch_metrics: Dictionary containing metrics for all epochs
        train_results: Results from training phase
        val_results: Results from validation phase
    """
    epoch_metrics["train_losses"].append(train_results["loss"])
    epoch_metrics["val_losses"].append(val_results["loss"])
    epoch_metrics["train_metrics"].append(train_results["metrics"])
    epoch_metrics["val_metrics"].append(val_results["metrics"])


def _print_final_metrics(
    best_val_metrics: List[MetricsDict], logger: logging.Logger
) -> None:
    """Print the final metrics for each fold.

    Args:
        best_val_metrics: List of best validation metrics for each fold
        logger: Logger instance for output
    """
    logger.info("\nBest validation metrics for each fold:")
    for fold, metrics in enumerate(best_val_metrics, 1):
        logger.info(f"\nFold {fold}:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    # Calculate and print average metrics across folds.
    avg_metrics = {}
    for metric in best_val_metrics[0].keys():
        avg_metrics[metric] = np.mean([fold[metric] for fold in best_val_metrics])

    logger.info("\nAverage metrics across all folds:")
    for metric, value in avg_metrics.items():
        logger.info(f"Average {metric}: {value:.4f}")

    # Calculate and print standard deviation of metrics.
    std_metrics = {}
    for metric in best_val_metrics[0].keys():
        std_metrics[metric] = np.std([fold[metric] for fold in best_val_metrics])

    logger.info("\nMetric standard deviations across folds:")
    for metric, value in std_metrics.items():
        logger.info(f"{metric} std: {value:.4f}")

"""This module implements a training pipeline for deep learning models using PyTorch."""

from tqdm import tqdm
import logging
import copy
import time  # Keep time import if used elsewhere, not directly in this snippet
from typing import Dict, List, Tuple, Union, Optional
from collections import OrderedDict

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

from models import Transformer, VAE
from .util import DataAugmenter, AugmentationConfig, SiameseDataset

MetricsDict = Dict[str, float]
FoldMetrics = Dict[
    str, List
]  # Though this type isn't directly used as a variable annotation later.


# Helper to re-initialize model and optimizer for a new fold/run
def _reinitialize_model_and_optimizer(
    pristine_template_cpu: nn.Module,
    base_optimizer_instance: optim.Optimizer,
    device: str,
) -> Tuple[nn.Module, optim.Optimizer]:
    """Re-initializes a model and optimizer for a new fold or run.

    This function creates a deep copy of the provided CPU model template and moves
    it to the specified device. It then creates a new optimizer instance of the
    same type as `base_optimizer_instance`, initialized with the parameters of
    the new model. This ensures that each fold or run starts with a fresh model
    and optimizer state.

    Args:
        pristine_template_cpu (nn.Module): A copy of the model template in CPU memory.
        base_optimizer_instance (optim.Optimizer): An instance of the optimizer to use as a template.
        device (str): The device to which the model should be moved (e.g., 'cuda', 'cpu', 'mps').

    Returns:
        Tuple[nn.Module, optim.Optimizer]: A new model instance on the specified device and
        a new optimizer instance initialized with the model's parameters.
    """
    new_model_gpu = copy.deepcopy(pristine_template_cpu).to(device)

    # Filter out problematic parameters that might not be accepted by the optimizer constructor
    optimizer_defaults = base_optimizer_instance.defaults.copy()
    # Remove parameters that are typically internal and not constructor arguments
    problematic_params = {"decoupled_weight_decay", "step_size", "gamma"}
    filtered_defaults = {
        k: v for k, v in optimizer_defaults.items() if k not in problematic_params
    }

    new_optimizer = type(base_optimizer_instance)(
        new_model_gpu.parameters(), **filtered_defaults
    )
    return new_model_gpu, new_optimizer


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 100,
    patience: int = 20,
    n_splits: int = 5,
    n_runs: int = 30,
    is_augmented: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    val_loader: Optional[DataLoader] = None,
) -> Tuple[nn.Module, Dict]:
    """Trains a model. If val_loader is provided, it performs a single training run.
    Otherwise, it performs k-fold cross-validation with multiple independent runs.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function to use for training.
        optimizer (optim.Optimizer): Optimizer instance to use for training.
        num_epochs (int): Number of epochs to train each fold.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        n_splits (int): Number of splits for k-fold cross-validation (if val_loader is not provided).
        n_runs (int): Number of independent runs to perform (if val_loader is not provided).
        is_augmented (bool): Whether to apply data augmentation during training.
        device (str): Device to use for training ('cuda', 'cpu', 'mps').
        val_loader (Optional[DataLoader]): DataLoader for the validation dataset. If provided, k-fold CV is skipped.

    Returns:
        Tuple[nn.Module, Dict]: The trained model on the specified device and a dictionary of metrics.
    """
    logger = logging.getLogger(__name__)

    if val_loader:
        # If a validation loader is provided, we are in a single fold of an outer CV loop.
        # We should not perform our own CV or runs here.
        fold_results = _train_fold(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs,
            patience,
            device,
            logger,
        )
        # The return type of train_model is Tuple[nn.Module, Dict].
        # _train_fold returns a Dict. I need to load the best model state and return it.
        final_model = model
        if fold_results["best_model_state"] is not None:
            final_model.load_state_dict(fold_results["best_model_state"])
        # The metrics dict from _train_fold is what we want.
        metrics = fold_results["best_fold_metrics"]
        if (
            "best_val_predictions" in fold_results
            and fold_results["best_val_predictions"] is not None
        ):
            metrics["best_val_predictions"] = fold_results["best_val_predictions"]
        return final_model, metrics

    pristine_model_template_cpu = copy.deepcopy(model).cpu()

    train_data_augmenter = None
    if is_augmented:
        aug_config = (
            AugmentationConfig(  # Define your static AugmentationConfig parameters here
                enabled=True,
                num_augmentations=5,
                noise_enabled=True,
                shift_enabled=True,
                scale_enabled=True,
                noise_level=0.1,
                shift_range=0.1,
                scale_range=0.1,
            )
        )
        train_data_augmenter = DataAugmenter(aug_config)

    if n_splits == 1:
        return _train_single_split(
            pristine_model_template_cpu,  # Pass the CPU template
            train_loader,
            criterion,
            optimizer,  # Pass base optimizer as template
            num_epochs,
            patience,
            train_data_augmenter,
            device,
            logger,
            n_runs,
        )

    all_runs_metrics_accumulator = []
    best_overall_accuracy = float("-inf")
    best_overall_model_state_cpu = None

    logger.info(
        f"Starting {n_runs} independent runs of {n_splits}-fold cross validation"
    )

    for run_idx in range(n_runs):
        logger.info(f"\nStarting Run {run_idx + 1}/{n_runs}")

        dataset = train_loader.dataset
        all_labels = _extract_labels(dataset)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run_idx)

        run_best_val_metrics_list = []
        current_run_best_accuracy = float("-inf")
        current_run_best_model_state_cpu = None

        fold_model_gpu, fold_optimizer = None, None

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(dataset)), all_labels), 1
        ):
            logger.info(f"Run {run_idx + 1}, Fold {fold_idx}/{n_splits}")

            if fold_model_gpu:
                del fold_model_gpu
            if fold_optimizer:
                del fold_optimizer
            if device == "cuda":
                torch.cuda.empty_cache()

            fold_model_gpu, fold_optimizer = _reinitialize_model_and_optimizer(
                pristine_model_template_cpu, optimizer, device
            )

            # Optimize DataLoader settings based on device
            num_workers = 0 if device == "mps" else (2 if device == "cuda" else 0)
            fold_train_loader, fold_val_loader = _create_fold_loaders(
                dataset,
                train_idx,
                val_idx,
                train_loader.batch_size,
                num_workers=num_workers,
                pin_memory=(device == "cuda"),
            )

            if train_data_augmenter:
                fold_train_loader = train_data_augmenter.augment(fold_train_loader)

            fold_results = _train_fold(
                fold_model_gpu,
                fold_train_loader,
                fold_val_loader,
                criterion,
                fold_optimizer,
                num_epochs,
                patience,
                device,
                logger,
            )

            if fold_results["best_accuracy"] > current_run_best_accuracy:
                current_run_best_accuracy = fold_results["best_accuracy"]
                current_run_best_model_state_cpu = fold_results["best_model_state"]

            run_best_val_metrics_list.append(fold_results["best_fold_metrics"])

        # Clean up after last fold of the run
        if fold_model_gpu:
            del fold_model_gpu
        if fold_optimizer:
            del fold_optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

        all_runs_metrics_accumulator.append(
            {
                "best_accuracy": current_run_best_accuracy,
                "best_val_metrics": run_best_val_metrics_list,
            }
        )

        if current_run_best_accuracy > best_overall_accuracy:
            best_overall_accuracy = current_run_best_accuracy
            best_overall_model_state_cpu = current_run_best_model_state_cpu
            logger.info(
                f"New best overall accuracy: {best_overall_accuracy:.4f} (Run {run_idx + 1})"
            )

    averaged_metrics = _calculate_averaged_metrics(all_runs_metrics_accumulator, logger)

    final_model_on_device = copy.deepcopy(pristine_model_template_cpu).to(device)
    if best_overall_model_state_cpu:
        final_model_on_device.load_state_dict(best_overall_model_state_cpu)
    else:
        logger.warning(
            "No best_overall_model_state found. Returning model with initial template weights."
        )

    return final_model_on_device, averaged_metrics


def _calculate_averaged_metrics(
    all_runs_metrics_accumulator: List[Dict], logger: logging.Logger
) -> Dict:
    """Calculates averaged metrics across all runs.

    Args:
        all_runs_metrics_accumulator (List[Dict]): List of dictionaries containing metrics from each run
        logger (logging.Logger): Logger instance for logging information.

    Returns:
        Dict: A dictionary containing averaged metrics across all runs, including overall accuracy and detailed metrics.
    """
    if not all_runs_metrics_accumulator:
        logger.warning("No run metrics available to calculate averages.")
        return {"runs_summary": {}, "metrics_summary": {}}

    all_accuracies = [
        run["best_accuracy"]
        for run in all_runs_metrics_accumulator
        if "best_accuracy" in run and run["best_accuracy"] != float("-inf")
    ]

    metrics_collector = {}
    if (
        all_runs_metrics_accumulator
        and all_runs_metrics_accumulator[0].get("best_val_metrics")
        and all_runs_metrics_accumulator[0]["best_val_metrics"]
        and all_runs_metrics_accumulator[0]["best_val_metrics"][0]
    ):  # Check nested structure
        for metric_key in all_runs_metrics_accumulator[0]["best_val_metrics"][0].keys():
            metrics_collector[metric_key] = []
    else:
        logger.warning(
            "Best validation metrics are missing or not in expected format for averaging."
        )

    for run_metrics_item in all_runs_metrics_accumulator:
        if "best_val_metrics" in run_metrics_item:
            for fold_metrics_dict in run_metrics_item["best_val_metrics"]:
                for metric_key, value in fold_metrics_dict.items():
                    if metric_key in metrics_collector:
                        metrics_collector[metric_key].append(value)

    avg_metrics_summary = {
        "runs_summary": {
            "accuracy_mean": (
                np.mean(all_accuracies) if all_accuracies else float("nan")
            ),
            "accuracy_std": np.std(all_accuracies) if all_accuracies else float("nan"),
        },
        "metrics_summary": {},
    }

    for metric_key, values in metrics_collector.items():
        valid_values = [v for v in values if not np.isnan(v)]
        avg_metrics_summary["metrics_summary"][metric_key] = {
            "mean": np.mean(valid_values) if valid_values else float("nan"),
            "std": np.std(valid_values) if valid_values else float("nan"),
        }

    logger.info("\nAveraged metrics across all runs:")
    logger.info(
        f"Overall Accuracy: {avg_metrics_summary['runs_summary']['accuracy_mean']:.4f} ± {avg_metrics_summary['runs_summary']['accuracy_std']:.4f}"
    )

    logger.info("\nDetailed metrics summary:")
    for metric_key, stats in avg_metrics_summary["metrics_summary"].items():
        logger.info(f"{metric_key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return avg_metrics_summary


def _train_single_split(
    pristine_model_template_cpu: nn.Module,
    train_loader: DataLoader,  # Full dataset loader
    criterion: nn.Module,
    base_optimizer_instance: optim.Optimizer,  # Optimizer template
    num_epochs: int,
    patience: int,
    train_data_augmenter: Optional[DataAugmenter],  # Pass augmenter instance
    device: str,
    logger: logging.Logger,
    n_runs: int = 30,
) -> Tuple[nn.Module, Dict]:
    """Trains a model using a single split with multiple independent runs.

    Args:
        pristine_model_template_cpu (nn.Module): A copy of the model template in CPU memory.
        train_loader (DataLoader): DataLoader for the full training dataset.
        criterion (nn.Module): Loss function to use for training.
        base_optimizer_instance (optim.Optimizer): Optimizer instance to use as a template.
        num_epochs (int): Number of epochs to train each fold.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        train_data_augmenter (Optional[DataAugmenter]): Data augmenter instance for training
        device (str): Device to use for training ('cuda', 'cpu', 'mps').
        logger (logging.Logger): Logger instance for logging information.
        n_runs (int): Number of independent runs to perform.

    Returns:
        Tuple[nn.Module, Dict]: The trained model on the specified device and a dictionary of
    """
    all_runs_metrics_accumulator = []
    best_overall_accuracy = float("-inf")
    best_overall_model_state_cpu = None

    logger.info(f"Starting {n_runs} independent runs for single split training")

    current_run_model_gpu, current_run_optimizer = None, None

    dataset = train_loader.dataset  # Get the full dataset for splitting

    for run_idx in range(n_runs):
        logger.info(f"\nStarting Run {run_idx + 1}/{n_runs}")

        if current_run_model_gpu:
            del current_run_model_gpu
        if current_run_optimizer:
            del current_run_optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

        current_run_model_gpu, current_run_optimizer = (
            _reinitialize_model_and_optimizer(
                pristine_model_template_cpu, base_optimizer_instance, device
            )
        )

        # Optimize DataLoader settings based on device
        num_workers = 0 if device == "mps" else (2 if device == "cuda" else 0)

        all_labels = _extract_labels(dataset)
        skf_single = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_idx)
        train_idx, val_idx = next(skf_single.split(np.zeros(len(dataset)), all_labels))

        run_train_loader, run_val_loader = _create_fold_loaders(
            dataset,
            train_idx,
            val_idx,
            train_loader.batch_size,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )

        if run_idx == 0:  # Log sizes only for the first run
            logger.info(
                f"Training set size: {len(run_train_loader.dataset)}"
            )  # Use loader.dataset to get Subset size
            logger.info(f"Validation set size: {len(run_val_loader.dataset)}")

        if train_data_augmenter:
            run_train_loader = train_data_augmenter.augment(run_train_loader)

        run_results = _train_fold(
            current_run_model_gpu,
            run_train_loader,
            run_val_loader,
            criterion,
            current_run_optimizer,
            num_epochs,
            patience,
            device,
            logger,
        )

        current_run_best_accuracy = run_results["best_accuracy"]

        all_runs_metrics_accumulator.append(
            {
                "best_accuracy": current_run_best_accuracy,
                "best_val_metrics": [run_results["best_fold_metrics"]],
            }
        )

        if current_run_best_accuracy > best_overall_accuracy:
            best_overall_accuracy = current_run_best_accuracy
            best_overall_model_state_cpu = run_results["best_model_state"]
            logger.info(
                f"New best overall accuracy: {best_overall_accuracy:.4f} (Run {run_idx + 1})"
            )

    # Clean up after the last run
    if current_run_model_gpu:
        del current_run_model_gpu
    if current_run_optimizer:
        del current_run_optimizer
    if device == "cuda":
        torch.cuda.empty_cache()

    averaged_metrics = _calculate_averaged_metrics(all_runs_metrics_accumulator, logger)

    final_model_on_device = copy.deepcopy(pristine_model_template_cpu).to(device)
    if best_overall_model_state_cpu:
        final_model_on_device.load_state_dict(best_overall_model_state_cpu)
    else:
        logger.warning(
            "No best_overall_model_state found in single_split. Returning model with initial template weights."
        )

    return final_model_on_device, averaged_metrics


def transfer_learning(  # No significant changes for conciseness here, it's a distinct utility
    dataset_name: str,
    model_instance: Transformer,
    file_path: str = "transformer_checkpoint.pth",
) -> Transformer:
    """Transfers learning weights from a checkpoint to a model instance for a specific dataset.

    Args:
        dataset_name (str): Name of the dataset for which the model is being adapted.
        model_instance (Transformer): Instance of the Transformer model to adapt.
        file_path (str): Path to the checkpoint file containing pre-trained weights.

    Returns:
        Transformer: The model instance with adapted weights for the specified dataset.
    """
    logger = logging.getLogger(__name__)
    output_dims_map = {
        "species": 2,
        "oil_simple": 2,
        "part": 7,
        "oil": 7,
        "cross-species": 3,
    }
    if dataset_name not in output_dims_map:
        raise ValueError(
            f"Invalid dataset: {dataset_name}. Valid: {list(output_dims_map.keys())}"
        )

    try:
        checkpoint = torch.load(
            file_path, map_location=torch.device("cpu")
        )  # Load to CPU first
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {file_path}")
        raise

    output_dim = output_dims_map[dataset_name]

    # Adapt the final layer (assuming 'fc' is the name)
    # This part is model-specific; for a generic Transformer, the output layer name might vary.
    if "fc.weight" in checkpoint and "fc.bias" in checkpoint:
        if dataset_name in [
            "species",
            "oil_simple",
        ]:  # Example: truncate if new output_dim is smaller
            checkpoint["fc.weight"] = checkpoint["fc.weight"][:output_dim]
            checkpoint["fc.bias"] = checkpoint["fc.bias"][:output_dim]
        else:  # Example: re-initialize for other cases (or use a more sophisticated strategy)
            original_fc_shape = checkpoint["fc.weight"].shape
            checkpoint["fc.weight"] = nn.init.xavier_uniform_(
                torch.empty(output_dim, original_fc_shape[1])
            )
            checkpoint["fc.bias"] = torch.zeros(output_dim)

        # If model_instance's fc layer has a different dimension, direct loading might fail.
        # Re-initialize the model's fc layer to match new output_dim before loading state_dict if needed.
        if (
            hasattr(model_instance, "fc")
            and model_instance.fc.out_features != output_dim
        ):
            model_instance.fc = nn.Linear(model_instance.fc.in_features, output_dim)
            logger.info(
                f"Re-initialized model's fc layer for {output_dim} output features."
            )

    else:
        logger.warning(
            f"Transfer learning: 'fc.weight' or 'fc.bias' not found in checkpoint. Final layer adaptation skipped."
        )

    model_instance.load_state_dict(checkpoint, strict=False)
    logger.info(
        f"Transferred learning weights from {file_path} to model for dataset {dataset_name}."
    )
    return model_instance


def _process_label_item(label_item) -> int:
    """Processes a label item to ensure it is returned as an integer.

    Args:
        label_item: The label item to process, which can be a tensor, numpy array,
        or a scalar value.

    Returns:
        int: The processed label as an integer.
    """
    if isinstance(label_item, torch.Tensor):
        return (
            label_item.item() if label_item.numel() == 1 else label_item.argmax().item()
        )
    elif isinstance(label_item, np.ndarray):
        return label_item.item() if label_item.size == 1 else np.argmax(label_item)
    return int(label_item)


def _extract_labels(dataset: Dataset) -> np.ndarray:
    """Extracts labels from a dataset, handling both Subset and full Dataset cases.

    Args:
        dataset (Dataset): The dataset from which to extract labels, can be a Subset or full Dataset.

    Returns:
        np.ndarray: An array of labels extracted from the dataset.
    """
    labels_list = []
    target_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = (
        dataset.indices if isinstance(dataset, Subset) else range(len(target_dataset))
    )

    if not indices:
        return np.array([])

    for i in indices:
        _, label_data = target_dataset[i]
        labels_list.append(label_data)

    return (
        np.array([_process_label_item(lbl) for lbl in labels_list])
        if labels_list
        else np.array([])
    )


def _create_fold_loaders(
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Creates DataLoaders for training and validation subsets based on provided indices.

    Args:
        dataset (Dataset): The full dataset from which to create subsets.
        train_idx (np.ndarray): Indices for the training subset.
        val_idx (np.ndarray): Indices for the validation subset.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker threads for loading data.
        pin_memory (bool): Whether to pin memory for faster data transfer to GPU.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for training and validation subsets.
    """
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    common_loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return (
        DataLoader(train_subset, shuffle=True, **common_loader_params),
        DataLoader(val_subset, shuffle=False, **common_loader_params),
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
    """Trains a model for a single fold of cross-validation.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function to use for training.
        optimizer (optim.Optimizer): Optimizer instance to use for training.
        num_epochs (int): Number of epochs to train the fold.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        device (str): Device to use for training ('cuda', 'cpu', 'mps').
        logger (logging.Logger): Logger instance for logging information.

    Returns:
        Dict: A dictionary containing the best validation accuracy, the best model state,
        the best fold metrics, and a log of metrics for each epoch.
    """
    best_val_accuracy = float("-inf")
    epochs_no_improve = 0
    best_model_state_cpu, best_fold_metrics, best_val_predictions = None, None, None
    epoch_log = {
        "train_losses": [],
        "val_losses": [],
        "train_metrics": [],
        "val_metrics": [],
    }

    for epoch in tqdm(
        range(num_epochs), desc="Fold Training", unit="epoch", leave=False
    ):
        model.train()
        train_results = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_training=True
        )

        model.eval()
        with torch.no_grad():
            val_results = _run_epoch(
                model, val_loader, criterion, None, device, is_training=False
            )

        epoch_log["train_losses"].append(train_results["loss"])
        epoch_log["val_losses"].append(val_results["loss"])
        epoch_log["train_metrics"].append(train_results["metrics"])
        epoch_log["val_metrics"].append(val_results["metrics"])

        current_val_acc = val_results["metrics"].get("balanced_accuracy", float("-inf"))

        if current_val_acc > best_val_accuracy:
            best_val_accuracy = current_val_acc
            best_model_state_cpu = OrderedDict(
                (k, v.clone().cpu()) for k, v in model.state_dict().items()
            )
            best_fold_metrics = {
                "train_loss": train_results["loss"],
                "train_accuracy": train_results["metrics"].get(
                    "balanced_accuracy", float("nan")
                ),
                "val_loss": val_results["loss"],
                "val_accuracy": val_results["metrics"].get(
                    "balanced_accuracy", float("nan")
                ),
                "epoch": epoch,
            }
            best_val_predictions = val_results["predictions"]
            epochs_no_improve = 0
            logger.debug(f"E{epoch+1}: New best val_acc: {best_val_accuracy:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping: E{epoch+1}, Best val_acc: {best_val_accuracy:.4f}"
                )
                break

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:  # Log less frequently
            logger.info(
                f"E{epoch+1} TL:{train_results['loss']:.3f} VL:{val_results['loss']:.3f} "
                f"VAcur:{current_val_acc:.3f} BestVAcur:{best_val_accuracy:.3f}"
            )

    # Ensure defaults if no training happened or no improvement
    if best_fold_metrics is None:
        # If no best model was found (e.g., due to early stopping before any improvement)
        # use the metrics from the last epoch, or empty dict if no epochs ran.
        if epoch_log["train_losses"]:
            best_fold_metrics = {
                "train_loss": epoch_log["train_losses"][-1],
                "train_accuracy": epoch_log["train_metrics"][-1].get(
                    "balanced_accuracy", float("nan")
                ),
                "val_loss": epoch_log["val_losses"][-1],
                "val_accuracy": epoch_log["val_metrics"][-1].get(
                    "balanced_accuracy", float("nan")
                ),
                "epoch": epoch,  # Use the last epoch number
            }
        else:
            best_fold_metrics = {
                "train_loss": float("nan"),
                "train_accuracy": float("nan"),
                "val_loss": float("nan"),
                "val_accuracy": float("nan"),
                "epoch": float("nan"),
            }
    if best_model_state_cpu is None:
        best_model_state_cpu = OrderedDict(
            (k, v.clone().cpu()) for k, v in model.state_dict().items()
        )

    return {
        "best_accuracy": (
            best_val_accuracy if best_val_accuracy > float("-inf") else 0.0
        ),
        "best_model_state": best_model_state_cpu,
        "best_fold_metrics": best_fold_metrics,
        "epoch_metrics": epoch_log,
        "best_val_predictions": best_val_predictions,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict:
    """Evaluates a model on a given data loader."""
    model.eval()
    with torch.no_grad():
        results = _run_epoch(model, loader, criterion, None, device, is_training=False)
    return results


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: str,
    is_training: bool,
) -> Dict:
    """Runs a single epoch of training or validation."""
    total_loss, all_labels_np, all_preds_np, all_probs_np = 0.0, [], [], []

    for batch in loader:
        # Unpack based on dataset type
        if isinstance(loader.dataset, SiameseDataset) or (
            isinstance(loader.dataset, Subset)
            and isinstance(loader.dataset.dataset, SiameseDataset)
        ):
            inputs1, inputs2, labels_batch = batch
            inputs1, inputs2, labels_on_device = (
                inputs1.to(device),
                inputs2.to(device),
                labels_batch.to(device),
            )
        else:
            inputs, labels_batch = batch
            inputs, labels_on_device = inputs.to(device), labels_batch.to(device)

        if is_training:
            optimizer.zero_grad()

        # Model specific forward pass
        if isinstance(loader.dataset, SiameseDataset) or (
            isinstance(loader.dataset, Subset)
            and isinstance(loader.dataset.dataset, SiameseDataset)
        ):
            outputs = model(inputs1, inputs2)
        else:
            outputs = model(inputs)

        # Prepare labels for loss and metrics
        if (
            labels_on_device.dim() > 1 and labels_on_device.shape[1] > 1
        ):  # one-hot to index
            actual_indices = labels_on_device.argmax(dim=1)
        elif labels_on_device.dim() > 1:  # [batch, 1] to [batch]
            actual_indices = labels_on_device.squeeze(-1)
        else:
            actual_indices = labels_on_device

        loss = criterion(
            outputs, actual_indices.long()
        )  # Ensure long type for CrossEntropy
        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * (
            inputs1.size(0)
            if isinstance(loader.dataset, SiameseDataset)
            or (
                isinstance(loader.dataset, Subset)
                and isinstance(loader.dataset.dataset, SiameseDataset)
            )
            else inputs.size(0)
        )

        # Process labels and predictions for metrics
        probs = torch.softmax(outputs, dim=1)
        predicted_indices = outputs.argmax(dim=1)

        all_labels_np.append(actual_indices.cpu().numpy())
        all_preds_np.append(predicted_indices.cpu().numpy())
        all_probs_np.append(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    final_labels = np.concatenate(all_labels_np)
    final_preds = np.concatenate(all_preds_np)
    final_probs = np.concatenate(all_probs_np)

    metrics = _calculate_metrics(final_labels, final_preds, final_probs)
    return {
        "loss": avg_loss,
        "metrics": metrics,
        "predictions": {
            "labels": final_labels,
            "preds": final_preds,
            "probs": final_probs,
        },
    }


def _calculate_metrics(  # Minor cleanup for NaN handling
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> MetricsDict:
    """Calculates various metrics based on true labels, predicted labels, and predicted probabilities.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (Optional[np.ndarray]): Predicted probabilities for each class, if available.

    Returns:
        MetricsDict: A dictionary containing calculated metrics such as balanced accuracy, precision, recall, F
    """
    labels_for_scoring = np.unique(np.concatenate([y_true, y_pred])).astype(int)
    metrics: MetricsDict = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
            labels=labels_for_scoring,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
            labels=labels_for_scoring,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
            labels=labels_for_scoring,
        ),
    }
    if (
        y_prob is not None and y_true.size > 0 and len(np.unique(y_true)) > 0
    ):  # Basic checks for valid AUC calculation
        n_classes = y_prob.shape[1]
        if n_classes == 2:
            metrics["auc_roc"] = roc_curve_auc(y_true, y_prob[:, 1])
        elif n_classes > 2:
            y_true_onehot = np.eye(n_classes)[y_true.astype(int)]
            aucs = [
                roc_curve_auc(
                    y_true_onehot[:, i],
                    y_prob[:, i],
                    class_present=(i in np.unique(y_true)),
                )
                for i in range(n_classes)
            ]
            valid_aucs = [a for a in aucs if not np.isnan(a)]
            metrics["auc_roc"] = np.mean(valid_aucs) if valid_aucs else float("nan")
        else:
            metrics["auc_roc"] = float("nan")  # Should not happen with >0 classes
    else:
        metrics["auc_roc"] = float("nan")
    return metrics


def roc_curve_auc(
    y_true_class: np.ndarray, y_prob_class: np.ndarray, class_present: bool = True
) -> float:
    """Calculates the AUC-ROC for a specific class.

    Args:
        y_true_class (np.ndarray): True labels for the specific class.
        y_prob_class (np.ndarray): Predicted probabilities for the specific class.
        class_present (bool): Whether the class is present in the true labels.

    Returns:
        float: The AUC-ROC value for the class, or NaN if the class is not present or not enough classes are available.
    """
    if (
        not class_present or len(np.unique(y_true_class)) < 2
    ):  # Not enough classes or class not present
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true_class, y_prob_class)
    return auc(fpr, tpr)

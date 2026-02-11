"""
This module implements a training pipeline for deep learning models using PyTorch.

It provides core functions for training models (`train_model`), running individual epochs (`_run_epoch`),
and evaluating models (`evaluate_model`). It supports standard training, cross-validation, and
sequential transfer learning workflows.
"""

from tqdm import tqdm
import logging
import copy
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold

from fishy.models.deep.transformer import Transformer
from fishy.data.augmentation import AugmentationConfig, DataAugmenter
from fishy.engine.trainer import Trainer
from fishy._core.utils import get_device

MetricsDict = Dict[str, float]
FoldMetrics = Dict[str, List]


def transfer_learning(
    dataset_name: str,
    model_instance: Transformer,
    file_path: str = "transformer_checkpoint.pth",
) -> Transformer:
    """
    Transfers learning weights from a checkpoint to a model instance for a specific dataset.

    This function loads pre-trained weights and adapts the final fully connected layer
    to match the output dimension of the target dataset.

    Args:
        dataset_name (str): Name of the dataset for which the model is being adapted.
        model_instance (Transformer): Instance of the Transformer model to adapt.
        file_path (str, optional): Path to the checkpoint file. Defaults to "transformer_checkpoint.pth".

    Returns:
        Transformer: The model instance with adapted weights for the specified dataset.

    Raises:
        ValueError: If the dataset name is invalid.
        FileNotFoundError: If the checkpoint file does not exist.
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
    if "fc.weight" in checkpoint and "fc.bias" in checkpoint:
        if dataset_name in [
            "species",
            "oil_simple",
        ]:  # Example: truncate if new output_dim is smaller
            checkpoint["fc.weight"] = checkpoint["fc.weight"][:output_dim]
            checkpoint["fc.bias"] = checkpoint["fc.bias"][:output_dim]
        else:  # Example: re-initialize for other cases
            original_fc_shape = checkpoint["fc.weight"].shape
            checkpoint["fc.weight"] = nn.init.xavier_uniform_(
                torch.empty(output_dim, original_fc_shape[1])
            )
            checkpoint["fc.bias"] = torch.zeros(output_dim)

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


def _reinitialize_model_and_optimizer(
    pristine_template_cpu: nn.Module,
    base_optimizer_instance: optim.Optimizer,
    device: str,
) -> Tuple[nn.Module, optim.Optimizer]:
    """
    Re-initializes a model and optimizer for a new fold or run.

    Args:
        pristine_template_cpu (nn.Module): A copy of the model template in CPU memory.
        base_optimizer_instance (optim.Optimizer): An instance of the optimizer to use as a template.
        device (str): The device to which the model should be moved.

    Returns:
        Tuple[nn.Module, optim.Optimizer]: A new model instance on the specified device and
        a new optimizer instance initialized with the model's parameters.
    """
    new_model_gpu = copy.deepcopy(pristine_template_cpu).to(device)

    optimizer_defaults = base_optimizer_instance.defaults.copy()
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
    device: Union[str, torch.device] = get_device(),
    val_loader: Optional[DataLoader] = None,
    use_coral: bool = False,
    use_cumulative_link: bool = False,
    num_classes: Optional[int] = None,
    regression: bool = False,
    ctx: Optional[Any] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Orchestrates the training process for a Deep Learning model.

    Supports two modes:
    1.  **Single Run**: If `val_loader` is provided, trains the model once on `train_loader`
        and evaluates on `val_loader`.
    2.  **Cross-Validation**: If `val_loader` is None, performs `n_splits`-fold Stratified
        Cross-Validation, repeated `n_runs` times.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data (or full dataset for CV).
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer instance (used as a template for resets).
        num_epochs (int, optional): Maximum training epochs. Defaults to 100.
        patience (int, optional): Early stopping patience. Defaults to 20.
        n_splits (int, optional): Number of CV folds. Defaults to 5.
        n_runs (int, optional): Number of independent CV runs. Defaults to 30.
        is_augmented (bool, optional): Whether to apply data augmentation. Defaults to False.
        device (Union[str, torch.device], optional): Computation device. Defaults to get_device().
        val_loader (Optional[DataLoader], optional): Explicit validation loader. Defaults to None.
        use_coral (bool, optional): Enable CORAL ordinal loss. Defaults to False.
        use_cumulative_link (bool, optional): Enable Cumulative Link ordinal loss. Defaults to False.
        num_classes (Optional[int], optional): Number of classes (for ordinal/coral). Defaults to None.
        regression (bool, optional): Enable regression mode. Defaults to False.
        ctx (Optional[Any], optional): RunContext for logging/tracking. Defaults to None.

    Returns:
        Tuple[nn.Module, Dict]: The best trained model and a dictionary of metrics.
    """
    logger = logging.getLogger(__name__)
    
    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)

    if val_loader:
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
            use_coral,
            use_cumulative_link,
            num_classes,
            regression,
            ctx=ctx,
        )
        final_model = model
        if fold_results["best_model_state"] is not None:
            final_model.load_state_dict(fold_results["best_model_state"])
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
        aug_config = AugmentationConfig(
            enabled=True,
            num_augmentations=5,
            noise_enabled=True,
            shift_enabled=True,
            scale_enabled=True,
            noise_level=0.1,
            shift_range=0.1,
            scale_range=0.1,
        )
        train_data_augmenter = DataAugmenter(aug_config)

    if n_splits == 1:
        return _train_single_split(
            pristine_model_template_cpu,
            train_loader,
            criterion,
            optimizer,
            num_epochs,
            patience,
            train_data_augmenter,
            device,
            logger,
            n_runs,
            use_coral,
            use_cumulative_link,
            num_classes,
            regression,
            ctx=ctx,
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
            if device.type == "cuda":
                torch.cuda.empty_cache()

            fold_model_gpu, fold_optimizer = _reinitialize_model_and_optimizer(
                pristine_model_template_cpu, optimizer, str(device)
            )

            num_workers = 0 if device.type == "mps" else (2 if device.type == "cuda" else 0)
            fold_train_loader, fold_val_loader = _create_fold_loaders(
                dataset,
                train_idx,
                val_idx,
                train_loader.batch_size,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
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
                use_coral,
                use_cumulative_link,
                num_classes,
                regression,
                ctx=ctx,
            )

            if fold_results["best_accuracy"] > current_run_best_accuracy:
                current_run_best_accuracy = fold_results["best_accuracy"]
                current_run_best_model_state_cpu = fold_results["best_model_state"]

            run_best_val_metrics_list.append(fold_results["best_fold_metrics"])

        if fold_model_gpu:
            del fold_model_gpu
        if fold_optimizer:
            del fold_optimizer
        if device.type == "cuda":
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
    """Calculates averaged metrics across all runs."""
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
    ):
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
        f"Overall Accuracy: {avg_metrics_summary['runs_summary']['accuracy_mean']:.4f} \u00b1 {avg_metrics_summary['runs_summary']['accuracy_std']:.4f}"
    )

    logger.info("\nDetailed metrics summary:")
    for metric_key, stats in avg_metrics_summary["metrics_summary"].items():
        logger.info(f"{metric_key}: {stats['mean']:.4f} \u00b1 {stats['std']:.4f}")

    return avg_metrics_summary


def _train_single_split(
    pristine_model_template_cpu: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    base_optimizer_instance: optim.Optimizer,
    num_epochs: int,
    patience: int,
    train_data_augmenter: Optional[DataAugmenter],
    device: torch.device,
    logger: logging.Logger,
    n_runs: int = 30,
    use_coral: bool = False,
    use_cumulative_link: bool = False,
    num_classes: Optional[int] = None,
    regression: bool = False,
    ctx: Optional[Any] = None,
) -> Tuple[nn.Module, Dict]:
    """Trains a model using a single split with multiple independent runs."""
    all_runs_metrics_accumulator = []
    best_overall_accuracy = float("-inf")
    best_overall_model_state_cpu = None

    logger.info(f"Starting {n_runs} independent runs for single split training")

    current_run_model_gpu, current_run_optimizer = None, None

    dataset = train_loader.dataset

    for run_idx in range(n_runs):
        logger.info(f"\nStarting Run {run_idx + 1}/{n_runs}")

        if current_run_model_gpu:
            del current_run_model_gpu
        if current_run_optimizer:
            del current_run_optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

        current_run_model_gpu, current_run_optimizer = (
            _reinitialize_model_and_optimizer(
                pristine_model_template_cpu, base_optimizer_instance, str(device)
            )
        )

        num_workers = 0 if device.type == "mps" else (2 if device.type == "cuda" else 0)

        all_labels = _extract_labels(dataset)
        skf_single = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_idx)
        train_idx, val_idx = next(skf_single.split(np.zeros(len(dataset)), all_labels))

        run_train_loader, run_val_loader = _create_fold_loaders(
            dataset,
            train_idx,
            val_idx,
            train_loader.batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        if run_idx == 0:
            logger.info(f"Training set size: {len(run_train_loader.dataset)}")
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
            use_coral,
            use_cumulative_link,
            num_classes,
            regression,
            ctx=ctx,
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

    if current_run_model_gpu:
        del current_run_model_gpu
    if current_run_optimizer:
        del current_run_optimizer
    if device.type == "cuda":
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


def _process_label_item(label_item) -> int:
    """Processes a label item to ensure it is returned as an integer."""
    if isinstance(label_item, torch.Tensor):
        return (
            label_item.item() if label_item.numel() == 1 else label_item.argmax().item()
        )
    elif isinstance(label_item, np.ndarray):
        return label_item.item() if label_item.size == 1 else np.argmax(label_item)
    return int(label_item)


def _extract_labels(dataset: Dataset) -> np.ndarray:
    """Extracts labels from a dataset, handling both Subset and full Dataset cases."""
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
    """Creates DataLoaders for training and validation subsets based on provided indices."""
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
    device: torch.device,
    logger: logging.Logger,
    use_coral: bool = False,
    use_cumulative_link: bool = False,
    num_classes: Optional[int] = None,
    regression: bool = False,
    ctx: Optional[Any] = None,
) -> Dict:
    """Trains a model for a single fold of cross-validation using the Trainer class."""
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        use_coral=use_coral,
        use_cumulative_link=use_cumulative_link,
        num_classes=num_classes,
        regression=regression,
        ctx=ctx,
        logger=logger,
    )
    return trainer.train(train_loader, val_loader)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device] = get_device(),
    use_coral: bool = False,
    use_cumulative_link: bool = False,
    num_classes: Optional[int] = None,
    regression: bool = False,
) -> Dict:
    """
    Evaluates a model on a given data loader.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader containing evaluation data.
        criterion (nn.Module): The loss function.
        device (Union[str, torch.device], optional): Computation device. Defaults to get_device().
        use_coral (bool, optional): Enable CORAL ordinal loss. Defaults to False.
        use_cumulative_link (bool, optional): Enable Cumulative Link ordinal loss. Defaults to False.
        num_classes (Optional[int], optional): Number of classes. Defaults to None.
        regression (bool, optional): Enable regression mode. Defaults to False.

    Returns:
        Dict: A dictionary containing 'loss', 'metrics', and 'predictions'.
    """
    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)

    # Create a dummy optimizer since it's not needed for evaluation
    optimizer = optim.Adam(model.parameters())
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=1, # Not used for evaluation
        use_coral=use_coral,
        use_cumulative_link=use_cumulative_link,
        num_classes=num_classes,
        regression=regression,
    )
    return trainer.evaluate(loader)

# -*- coding: utf-8 -*-
"""
Main entry point for the deep learning model training pipeline.

This script orchestrates the pre-training and fine-tuning of various deep learning models
for spectral data analysis. It handles command-line argument parsing, configuration,
data loading, model creation, and the execution of training phases.

The pipeline is designed to be modular, allowing for different models, datasets, and
pre-training tasks to be specified via command-line arguments.

Example Usage:
--------------
# Fine-tune a transformer model on the 'species' dataset
python -m deep-learning.main --model transformer --dataset species --file-path /path/to/data/REIMS.xlsx

# Pre-train a model using masked spectra modelling, then fine-tune
python -m deep-learning.main --model transformer --dataset species --file-path /path/to/data/REIMS.xlsx \
    --masked-spectra-modelling --epochs 50

"""

# ## 1. Imports and Setup
# -----------------------
# Standard library and third-party imports.

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Callable, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from models import (
    Transformer,
    LSTM,
    CNN,
    RCNN,
    Mamba,
    KAN,
    VAE,
    MOE,
    Dense,
    ODE,
    RWKV,
    TCN,
    WaveNet,
    Ensemble,
    Diffusion,
)
from .pre_training import PreTrainer, PreTrainingConfig
from .train import train_model
from .util import create_data_module, AugmentationConfig, CustomDataset, SiameseDataset

# ## 2. Configuration
# --------------------
# Centralized configuration for the training pipeline.


@dataclass
class TrainingConfig:
    """
    Configuration for model training, combining settings from command-line arguments.
    """

    file_path: str
    model: str
    dataset: str
    run: int
    output: str
    data_augmentation: bool
    masked_spectra_modelling: bool
    next_spectra_prediction: bool
    next_peak_prediction: bool
    spectrum_denoising_autoencoding: bool
    peak_parameter_regression: bool
    spectrum_segment_reordering: bool
    contrastive_transformation_invariance_learning: bool
    early_stopping: int
    dropout: float
    label_smoothing: float
    epochs: int
    learning_rate: float
    batch_size: int
    hidden_dimension: int
    num_layers: int
    num_heads: int
    num_augmentations: int
    noise_level: float
    shift_enabled: bool
    scale_enabled: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create configuration from command line arguments."""
        return cls(**vars(args))


# ## 3. Model Factory
# --------------------
# A factory for creating model instances. This approach avoids long if/elif chains
# and makes it easy to register new models.

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "transformer": Transformer,
    "lstm": LSTM,
    "cnn": CNN,
    "rcnn": RCNN,
    "mamba": Mamba,
    "kan": KAN,
    "vae": VAE,
    "moe": MOE,
    "dense": Dense,
    "ode": ODE,
    "rwkv": RWKV,
    "tcn": TCN,
    "wavenet": WaveNet,
    "ensemble": Ensemble,
    "diffusion": Diffusion,
}


def create_model(config: TrainingConfig, input_dim: int, output_dim: int) -> nn.Module:
    """
    Creates a model instance based on the model specified in the config.

    Args:
        config: Training configuration containing model type and parameters.
        input_dim: Dimension of the input features.
        output_dim: Number of output classes.

    Returns:
        An instance of the specified model class initialized with the given dimensions
        and configuration parameters.
    """
    model_class = MODEL_REGISTRY.get(config.model)
    if not model_class:
        raise ValueError(f"Invalid model type: {config.model}")

    # Common arguments for most models
    model_args = {"dropout": config.dropout}

    # Model-specific argument mapping
    if config.model == "transformer":
        model_args.update(
            {
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "hidden_dim": config.hidden_dimension,
            }
        )
        return Transformer(input_dim=input_dim, output_dim=output_dim, **model_args)
    elif config.model == "ensemble":
        return Ensemble(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dimension,
            dropout=config.dropout,
            device=(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            ),
        )
    elif config.model == "lstm":
        model_args.update(
            {"hidden_size": config.hidden_dimension, "num_layers": config.num_layers}
        )
        return LSTM(input_size=input_dim, output_size=output_dim, **model_args)
    elif config.model in ["cnn", "rcnn"]:
        return model_class(input_size=input_dim, num_classes=output_dim, **model_args)
    elif config.model == "mamba":
        return Mamba(
            d_model=input_dim,
            n_classes=output_dim,
            d_state=config.hidden_dimension,
            d_conv=4,
            expand=2,
            depth=config.num_layers,
        )
    elif config.model == "kan":
        return KAN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dimension,
            num_layers=config.num_layers,
            dropout_rate=config.dropout,
            num_inner_functions=10,
        )
    elif config.model == "vae":
        return VAE(
            input_size=input_dim,
            num_classes=output_dim,
            latent_dim=config.hidden_dimension,
            **model_args,
        )
    elif config.model == "moe":
        return MOE(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dimension,
            num_experts=4,
            k=2,
        )
    elif config.model == "diffusion":
        return Diffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dimension,
            time_dim=64,
            num_timesteps=1000,
        )
    else:  # For models like Dense, ODE, RWKV, TCN, WaveNet, Ensemble
        return model_class(input_dim=input_dim, output_dim=output_dim, **model_args)


# ## 4. Training Orchestrator
# ----------------------------
# The ModelTrainer class manages the overall training process, including
# pre-training and fine-tuning.


class ModelTrainer:
    """
    Orchestrates the model training pipeline, from data setup to pre-training and fine-tuning.
    """

    N_CLASSES_PER_DATASET = {
        "species": 2,
        "part": 7,
        "oil": 7,
        "cross-species": 3,
        "cross-species-hard": 15,
        "instance-recognition": 2,
        "instance-recognition-hard": 24,
    }
    PRETRAIN_TASK_DEFINITIONS: List[
        Tuple[str, Callable[["ModelTrainer"], int], str, bool, Dict[str, Any]]
    ] = [
        (
            "masked_spectra_modelling",
            lambda self: self.n_features,
            "pre_train_masked_spectra",
            False,
            {},
        ),
        ("next_spectra_prediction", lambda self: 2, "pre_train_next_spectra", True, {}),
        (
            "next_peak_prediction",
            lambda self: self.n_features,
            "pre_train_peak_prediction",
            False,
            {"peak_threshold": 0.1, "window_size": 5},
        ),
        (
            "spectrum_denoising_autoencoding",
            lambda self: self.n_features,
            "pre_train_denoising_autoencoder",
            False,
            {},
        ),
        (
            "peak_parameter_regression",
            lambda self: self.n_features,
            "pre_train_peak_parameter_regression",
            False,
            {},
        ),
        (
            "spectrum_segment_reordering",
            lambda self: 4 * 4,
            "pre_train_spectrum_segment_reordering",
            False,
            {"num_segments": 4},
        ),
        (
            "contrastive_transformation_invariance_learning",
            lambda self: 128,
            "pre_train_contrastive_invariance",
            False,
            {"embedding_dim": 128},
        ),
    ]

    def __init__(self, config: TrainingConfig):
        """Initializes the ModelTrainer with the provided configuration.

        Args:
            config: Training configuration containing all necessary parameters.

        Raises:
            ValueError: If the specified dataset is not supported.
        """
        self.config = config
        self.logger = self._setup_logging()
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if self.config.dataset not in self.N_CLASSES_PER_DATASET:
            raise ValueError(f"Invalid dataset: {self.config.dataset}")
        self.n_classes = self.N_CLASSES_PER_DATASET[self.config.dataset]
        self.data_module = create_data_module(
            file_path=config.file_path,
            dataset_name=config.dataset,
            batch_size=config.batch_size,
            augmentation_config=config,
        )
        self.data_module.setup()
        self.n_features = self.data_module.get_input_dim()

    def _setup_logging(self) -> logging.Logger:
        """Configures logging to file and console.

        Creates a log file in the output directory with the run identifier.

        Returns:
            A configured logger instance.
        """
        log_file = (
            Path(self.config.output).parent
            / f"{Path(self.config.output).name}_{self.config.run}.log"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.addHandler(console_handler)
        logger.propagate = False
        return logger

    def pre_train(self) -> Optional[nn.Module]:
        """
        Executes the enabled pre-training tasks sequentially.

        Returns:
            The model after the last pre-training task, or None if no tasks are enabled.
        """
        self.logger.info("Evaluating pre-training phase")
        enabled_tasks = [
            task
            for task in self.PRETRAIN_TASK_DEFINITIONS
            if getattr(self.config, task[0], False)
        ]
        if not enabled_tasks:
            self.logger.info("No pre-training tasks enabled.")
            return None

        self.logger.info(
            f"Enabled pre-training tasks: {', '.join(t[0] for t in enabled_tasks)}"
        )
        if self.data_module is None:
            self.logger.error("Pre-training DataModule not set.")
            return None
        train_loader: DataLoader = self.data_module.get_train_dataloader()
        val_loader: Optional[DataLoader] = (
            self.data_module.get_val_dataloader()
            if hasattr(self.data_module, "get_val_dataloader")
            else None
        )

        pre_train_cfg = PreTrainingConfig(
            num_epochs=self.config.epochs,
            file_path=self.config.file_path,
            device=self.device,
            n_features=self.n_features,
        )

        model_after_last_task: Optional[nn.Module] = None
        for flag, out_dim_fn, method, req_val, kwargs in enabled_tasks:
            self.logger.info(f"Starting pre-training task: {flag}")
            output_dim = out_dim_fn(self)

            current_model = create_model(self.config, self.n_features, output_dim).to(
                self.device
            )
            if model_after_last_task:
                self._handle_weight_chaining(current_model, model_after_last_task)

            pre_trainer = PreTrainer(
                model=current_model,
                config=pre_train_cfg,
                optimizer=torch.optim.AdamW(
                    current_model.parameters(), lr=self.config.learning_rate
                ),
            )

            call_args = [train_loader]
            if req_val:
                if val_loader is None:
                    self.logger.warning(
                        f"Validation loader for {flag} not found, passing None."
                    )
                call_args.append(val_loader)

            start_time = time.time()
            trained_model = getattr(pre_trainer, method)(*call_args, **kwargs)
            self.logger.info(f"{flag} training time: {time.time() - start_time:.2f}s")
            model_after_last_task = trained_model

        self.logger.info("Pre-training completed.")
        return model_after_last_task

    def _handle_weight_chaining(self, current_model: nn.Module, prev_model: nn.Module):
        """
        Loads weights from a previously trained model into the current model,
        matching layers by name and shape.

        Args:
            current_model: The model to which weights will be loaded.
            prev_model: The previously trained model from which weights are taken.

        Raises:
            Exception: If the weight loading fails, a warning is logged and training continues from scratch.
        """
        self.logger.info(
            f"Attempting to load weights from previous model for {self.config.model}"
        )
        try:
            prev_state_dict = prev_model.state_dict()
            current_model_dict = current_model.state_dict()

            load_state_dict = {
                k: v
                for k, v in prev_state_dict.items()
                if k in current_model_dict and v.shape == current_model_dict[k].shape
            }
            missing_keys, unexpected_keys = current_model.load_state_dict(
                load_state_dict, strict=False
            )
            if missing_keys:
                self.logger.warning(f"Chaining: Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Chaining: Unexpected keys: {unexpected_keys}")
            self.logger.info("Weight chaining: successfully loaded compatible weights.")
        except Exception as e:
            self.logger.warning(
                f"Weight chaining failed: {e}. Model will train from scratch."
            )

    def train(self, pre_trained_model: Optional[nn.Module] = None) -> nn.Module:
        """
        Executes the main fine-tuning phase using Group k-fold cross-validation.

        Args:
            pre_trained_model: An optional pre-trained model to adapt for fine-tuning.

        Returns:
            The final trained model instance (from the last fold).
        """
        dataset_name_str = self.data_module.processor.dataset_type.name.lower().replace(
            "_", "-"
        )
        if "instance-recognition" in dataset_name_str:
            self.logger.info(
                "Starting main fine-tuning phase with Stratified Group K-Fold Cross-Validation"
            )
        else:
            self.logger.info(
                "Starting main fine-tuning phase with Stratified K-Fold Cross-Validation"
            )

        if self.data_module is None:
            self.logger.error("Fine-tuning DataModule not set.")
            return create_model(self.config, self.n_features, self.n_classes)

        # Get full dataset from DataModule
        full_dataset_samples = self.data_module.get_dataset().samples.cpu().numpy()
        full_dataset_labels = self.data_module.get_dataset().labels.cpu().numpy()

        k_folds = 3  # User requested k=3

        if "instance-recognition" in dataset_name_str:
            full_dataset_groups = self.data_module.processor.extract_groups(
                self.data_module.filtered_data
            )
            cv_splitter = StratifiedGroupKFold(n_splits=k_folds)
            split_args = (
                full_dataset_samples,
                np.argmax(full_dataset_labels, axis=1),
                full_dataset_groups,
            )
        else:
            cv_splitter = StratifiedKFold(
                n_splits=k_folds, shuffle=True, random_state=self.config.run
            )
            split_args = (full_dataset_samples, np.argmax(full_dataset_labels, axis=1))

        all_fold_metrics = []
        final_trained_model = None  # To store the model from the last fold

        for fold, (train_index, val_index) in enumerate(cv_splitter.split(*split_args)):
            self.logger.info(f"--- Starting Fold {fold + 1}/{k_folds} ---")

            X_train, X_val = (
                full_dataset_samples[train_index],
                full_dataset_samples[val_index],
            )
            y_train, y_val = (
                full_dataset_labels[train_index],
                full_dataset_labels[val_index],
            )

            # Determine dataset class (e.g. Siamese for instance recognition)
            dataset_name_str = (
                self.data_module.processor.dataset_type.name.lower().replace("_", "-")
            )
            dataset_class = (
                SiameseDataset
                if "instance-recognition" in dataset_name_str
                else CustomDataset
            )

            train_dataset = dataset_class(X_train, y_train)
            val_dataset = dataset_class(X_val, y_val)

            # Check for empty datasets or lack of positive pairs for Siamese
            if isinstance(train_dataset, SiameseDataset):
                num_train_pairs = len(train_dataset)
                num_train_pos_pairs = np.sum(
                    train_dataset.paired_labels.cpu().numpy() == 1
                )
                self.logger.info(
                    f"Train pairs: {num_train_pairs} (Positive: {num_train_pos_pairs})"
                )
                if num_train_pairs == 0 or num_train_pos_pairs == 0:
                    self.logger.warning(
                        f"Fold {fold + 1} train set has no pairs or no positive pairs. Skipping fold."
                    )
                    continue
            else:
                self.logger.info(f"Train samples: {len(train_dataset)}")

            if isinstance(val_dataset, SiameseDataset):
                num_val_pairs = len(val_dataset)
                num_val_pos_pairs = np.sum(val_dataset.paired_labels.cpu().numpy() == 1)
                self.logger.info(
                    f"Validation pairs: {num_val_pairs} (Positive: {num_val_pos_pairs})"
                )
                if num_val_pairs == 0 or num_val_pos_pairs == 0:
                    self.logger.warning(
                        f"Fold {fold + 1} validation set has no pairs or no positive pairs. Skipping fold."
                    )
                    continue
            else:
                self.logger.info(f"Validation samples: {len(val_dataset)}")

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )  # No shuffle for val

            model_to_finetune = create_model(
                self.config, self.n_features, self.n_classes
            ).to(self.device)

            if pre_trained_model:
                self.logger.info("Transferring pre-trained weights for fine-tuning.")
                self._adapt_pretrained_model_for_finetuning(
                    model_to_finetune, pre_trained_model
                )

            criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            optimizer = torch.optim.AdamW(
                model_to_finetune.parameters(), lr=self.config.learning_rate
            )

            trained_model_instance, metrics = train_model(
                model=model_to_finetune,
                train_loader=train_loader,
                val_loader=val_loader,  # Pass validation loader
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=self.config.epochs,
                patience=self.config.early_stopping,
                is_augmented=self.config.data_augmentation,
                device=self.device,
            )
            all_fold_metrics.append(metrics)
            final_trained_model = trained_model_instance  # Keep the last one

        self.logger.info("Cross-Validation finished.")

        if all_fold_metrics:
            # Aggregate and log cross-validation results
            avg_metrics = {
                k: np.mean([m[k] for m in all_fold_metrics])
                for k in all_fold_metrics[0]
            }
            self.logger.info(f"Average metrics across {k_folds} folds: {avg_metrics}")
            # You might want to save these aggregated metrics to a file
        else:
            self.logger.warning("No folds completed successfully to aggregate metrics.")

        return final_trained_model


# ## 5. Argument Parsing
# -----------------------
# Defines and parses command-line arguments.


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(
        prog="ModelTraining", description="Spectra Model Training Pipeline."
    )

    # Core arguments
    parser.add_argument(
        "-fp",
        "--file-path",
        type=str,
        default="/Users/woodj/Desktop/fishy-business/data/REIMS.xlsx",
        help="Path to the dataset file (e.g., REIMS.xlsx)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="species",
        choices=ModelTrainer.N_CLASSES_PER_DATASET.keys(),
        help="Dataset name",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="transformer",
        choices=MODEL_REGISTRY.keys(),
        help="Model type",
    )
    parser.add_argument(
        "-r", "--run", type=int, default=0, help="Run identifier for logging"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="logs/results_base",
        help="Base name for output logs",
    )

    # Pre-training task flags
    for task_flag, _, _, _, _ in ModelTrainer.PRETRAIN_TASK_DEFINITIONS:
        parser.add_argument(
            f"--{task_flag.replace('_', '-')}",
            action="store_true",
            help=f"Enable {task_flag.replace('_', ' ')} pre-training",
        )

    # Training hyperparameters
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )

    # Regularization
    parser.add_argument(
        "-es", "--early-stopping", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "-do", "--dropout", type=float, default=0.2, help="Dropout probability"
    )
    parser.add_argument(
        "-ls",
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing alpha",
    )

    # Model architecture
    parser.add_argument(
        "-hd", "--hidden-dimension", type=int, default=128, help="Hidden dimension size"
    )
    parser.add_argument(
        "-l", "--num-layers", type=int, default=4, help="Number of layers"
    )
    parser.add_argument(
        "-nh",
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads (for Transformer)",
    )

    # Data Augmentation
    parser.add_argument(
        "-da",
        "--data-augmentation",
        action="store_true",
        help="Enable data augmentation",
    )
    parser.add_argument(
        "--num-augmentations", type=int, default=5, help="Augmentations per sample"
    )
    parser.add_argument(
        "--noise-level", type=float, default=0.1, help="Noise level for augmentation"
    )
    parser.add_argument(
        "--shift-enabled", action="store_true", help="Enable shift augmentation"
    )
    parser.add_argument(
        "--scale-enabled", action="store_true", help="Enable scale augmentation"
    )

    return parser.parse_args()


# ## 6. Main Execution
# ---------------------
# The main function that ties everything together.


def main() -> None:
    """Main execution function.

    Raises:
        Exception: If any critical error occurs during the training pipeline, it logs the error and re-raises it.
    """
    trainer_instance = None
    try:
        args = parse_arguments()
        config = TrainingConfig.from_args(args)

        trainer_instance = ModelTrainer(config)
        logger = trainer_instance.logger
        logger.info(f"Training configuration: {config}")
        logger.info(f"Using device: {trainer_instance.device}")
        logger.info("Starting training pipeline")

        # --- Pre-training Phase ---
        any_pretrain_task_enabled = any(
            getattr(config, task[0]) for task in ModelTrainer.PRETRAIN_TASK_DEFINITIONS
        )
        pre_trained_model = None
        if any_pretrain_task_enabled:
            pre_trained_model = trainer_instance.pre_train()
        else:
            logger.info("Skipping pre-training phase.")

        # --- Fine-tuning Phase ---
        final_trained_model = trainer_instance.train(pre_trained_model)
        logger.info(
            f"Training pipeline completed. Final model: {type(final_trained_model).__name__}"
        )

    except Exception as e:
        # Use the instance's logger if available, otherwise use a default logger.
        current_logger = getattr(
            trainer_instance, "logger", logging.getLogger(__name__)
        )
        current_logger.error(f"Critical error in training pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

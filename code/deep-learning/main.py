import argparse
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from collections import defaultdict
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import StratifiedKFold
import numpy as np

from plot import plot_attention_map
from pre_training import PreTrainer, PreTrainingConfig
from lstm import LSTM
from transformer import Transformer
from cnn import CNN
from rcnn import RCNN
from mamba import Mamba
from kan import KAN
from vae import VAE
from MOE import MOE
from dense import Dense
from ode import ODE
from rwkv import RWKV
from tcn import TCN
from wavenet import WaveNet
from ensemble import Ensemble
from diffusion import Diffusion

from train import train_model
from util import preprocess_dataset, create_data_module, AugmentationConfig

@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # File and dataset settings
    file_path: str = "transformer_checkpoint.pth"
    model_type: str = "transformer"
    dataset: str = "species"
    run: int = 0
    output: str = "logs/results"

    # Preprocessing flags
    data_augmentation: bool = False
    masked_spectra_modelling: bool = False
    next_spectra_prediction: bool = False

    # Regularization settings
    early_stopping: int = 10
    dropout: float = 0.2
    label_smoothing: float = 0.1

    # Training hyperparameters
    epochs: int = 100
    learning_rate: float = 1e-5
    batch_size: int = 64
    hidden_dimension: int = 128
    num_layers: int = 4
    num_heads: int = 4

    # Data augmentation settings
    num_augmentations: int = 5
    noise_level: float = 0.1
    shift_enabled: bool = False
    scale_enabled: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create configuration from command line arguments."""
        return cls(
            file_path=args.file_path,
            dataset=args.dataset,
            model_type=args.model,
            run=args.run,
            output=args.output,
            data_augmentation=args.data_augmentation,
            masked_spectra_modelling=args.masked_spectra_modelling,
            next_spectra_prediction=args.next_spectra_prediction,
            early_stopping=args.early_stopping,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            hidden_dimension=args.hidden_dimension,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_augmentations=args.num_augmentations,
            noise_level=args.noise_level,
            shift_enabled=args.shift_enabled,
            scale_enabled=args.scale_enabled,
        )


class ModelTrainer:
    """Handles the complete training pipeline including pre-training and fine-tuning."""

    N_CLASSES_PER_DATASET = {
        "species": 2,
        "part": 7,
        "oil": 7,
        "cross-species": 3,
        "cross-species-hard": 15,
        "instance-recognition": 2,
        "instance-recognition-hard": 24,
    }

    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.logger = self._setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = config.model_type

        if config.dataset not in self.N_CLASSES_PER_DATASET:
            raise ValueError(f"Invalid dataset: {config.dataset}")

        self.n_classes = self.N_CLASSES_PER_DATASET[config.dataset]
        self.n_features = 2080 # Could be made configurable if needed

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        output = f"{self.config.output}_{self.config.run}.log"
        logging.basicConfig(
            filename=output,
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        return logger

    def pre_train(self) -> Optional[Transformer]:
        """Run pre-training if configured."""
        if not (
            self.config.masked_spectra_modelling or self.config.next_spectra_prediction
        ):
            return None

        self.logger.info("Starting pre-training phase")

        # Load pre-training data
        train_loader, data = self.data_module.setup()

        # Initialize model for pre-training
        model = self._create_model(
            input_dim=self.n_features, output_dim=self.n_features
        )
        model = model.to(self.device)

        # Setup pre-training configuration
        pre_training_config = PreTrainingConfig(
            num_epochs=self.config.epochs,
            file_path=self.config.file_path,
            device=self.device,
        )

        # Initialize pre-trainer
        pre_trainer = PreTrainer(
            model=model,
            config=pre_training_config,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.AdamW(
                model.parameters(), lr=self.config.learning_rate
            ),
        )

        # Masked Spectra Modelling
        if self.config.masked_spectra_modelling:
            # self.logger.info("Starting Masked Spectra Modelling")
            # start_time = time.time()
            # model = pre_trainer.pre_train_masked_spectra(train_loader)
            # self.logger.info(f"MSM training time: {time.time() - start_time:.2f}s")

            self.logger.info("Starting Peak Prediction")
            model = pre_trainer.pre_train_peak_prediction(
                train_loader,
                peak_threshold=0.1,  # Adjust based on your data
                window_size=5        # Adjust based on peak width
            )

        # Next Spectra Prediction
        if self.config.next_spectra_prediction:
            self.logger.info("Starting Next Spectra Prediction")
            start_time = time.time()
            # Reinitialize model for NSP if needed
            if self.config.masked_spectra_modelling:
                model = self._create_model(
                    input_dim=self.n_features, output_dim=self.n_features
                )
                pre_trainer = PreTrainer(
                    model=model,
                    config=pre_training_config,
                    criterion=nn.CrossEntropyLoss(
                        label_smoothing=self.config.label_smoothing
                    ),
                    optimizer=torch.optim.AdamW(
                        model.parameters(), lr=self.config.learning_rate
                    ),
                )

            val_loader, _ = preprocess_dataset(
                self.config.dataset,
                self.config.data_augmentation,
                batch_size=self.config.batch_size,
                is_pre_train=True,
            )
            model = pre_trainer.pre_train_next_spectra(train_loader, val_loader)
            self.logger.info(f"NSP training time: {time.time() - start_time:.2f}s")

        return model

    def _calculate_metrics(self, true_labels, predictions) -> Dict[str, float]:
        """Calculate metrics with proper multi-class handling."""
        # Ensure inputs are numpy arrays
        true_labels = np.asarray(true_labels)
        predictions = np.asarray(predictions)

        if len(true_labels) != len(predictions):
            raise ValueError(f"Inconsistent number of samples: true_labels={len(true_labels)}, predictions={len(predictions)}")

        # Calculate metrics with proper multi-class handling
        accuracy = balanced_accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)

        # Add per-class metrics for debugging
        class_precisions = precision_score(true_labels, predictions, average=None, zero_division=0)
        class_recalls = recall_score(true_labels, predictions, average=None, zero_division=0)
        class_f1s = f1_score(true_labels, predictions, average=None, zero_division=0)

        # Calculate class distribution
        class_dist = np.bincount(true_labels, minlength=self.n_classes)
        pred_dist = np.bincount(predictions, minlength=self.n_classes)

        print("\nPer-class metrics:")
        for i in range(self.n_classes):
            print(f"Class {i}:")
            print(f"  True count: {class_dist[i]}, Predicted count: {pred_dist[i]}")
            print(f"  Precision: {class_precisions[i]:.4f}")
            print(f"  Recall: {class_recalls[i]:.4f}")
            print(f"  F1: {class_f1s[i]:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _evaluate_fold(
        self, model: nn.Module, val_loader: DataLoader
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Evaluate a single fold with and without test-time compute."""
        outputs_standard = []
        outputs_ttc = []
        outputs_gp = []
        outputs_mc = []
        labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Standard forward pass
                out_std = model(x)
                outputs_standard.append(out_std)
                labels.append(y)  # Only append labels once

                outputs_ttc.append(out_std)  # Use standard output
                outputs_gp.append(out_std)   # Use standard output

        # Concatenate all outputs and labels
        outputs_standard = torch.cat(outputs_standard, dim=0)
        outputs_ttc = torch.cat(outputs_ttc, dim=0)
        outputs_gp = torch.cat(outputs_gp, dim=0)
        outputs_mc = torch.cat(outputs_mc, dim=0)
        labels = torch.cat(labels, dim=0)

        # Debug: Print unique predictions for each method
        print("\nUnique predictions per method:")
        print(f"Standard: {np.unique(outputs_standard.argmax(dim=-1).cpu().numpy())}")
        print(f"Beam Search: {np.unique(outputs_ttc.argmax(dim=-1).cpu().numpy())}")
        print(f"Genetic: {np.unique(outputs_gp.argmax(dim=-1).cpu().numpy())}")
        print(f"Monte Carlo: {np.unique(outputs_mc.argmax(dim=-1).cpu().numpy())}")

        # Debug: Print confidence distributions
        print("\nConfidence statistics:")
        for name, outputs in [('Standard', outputs_standard), ('Beam', outputs_ttc), ('Genetic', outputs_gp)]:
            probs = torch.softmax(outputs, dim=-1)
            max_probs = probs.max(dim=-1)[0]
            print(f"{name} - Mean: {max_probs.mean():.3f}, Std: {max_probs.std():.3f}, "
                f"Min: {max_probs.min():.3f}, Max: {max_probs.max():.3f}")

        # Convert to numpy for metric calculation
        pred_std = outputs_standard.argmax(dim=-1).cpu().numpy()
        pred_ttc = outputs_ttc.argmax(dim=-1).cpu().numpy()
        pred_gp = outputs_gp.argmax(dim=-1).cpu().numpy()
        pred_mc = outputs_mc.argmax(dim=-1).cpu().numpy()
        true_labels = labels.argmax(dim=-1).cpu().numpy()

        # Calculate metrics
        metrics_std = self._calculate_metrics(true_labels, pred_std)
        metrics_ttc = self._calculate_metrics(true_labels, pred_ttc)
        metrics_gp = self._calculate_metrics(true_labels, pred_gp)
        metrics_mc = self._calculate_metrics(true_labels, pred_mc)

        return metrics_std, metrics_ttc, metrics_gp, metrics_mc

    def train(self, pre_trained_model: Optional[Transformer] = None) -> Transformer:
        """Run main training phase with improved PRM training and evaluation."""
        self.logger.info("Starting main training phase")

        # Load training data
        train_loader, data = self.data_module.setup()

        # Initialize model
        model = self._create_model(self.n_features, self.n_classes)
        if pre_trained_model is not None:
            self.logger.info("Transferring pre-trained weights")

            # Get the checkpoint from the pretrained model.
            checkpoint = pre_trained_model.state_dict()

            print(f"checkpoint keys: {checkpoint.keys()}")

            # Adjust the weights and bias for the final layer.
            checkpoint["fc_out.weight"] = checkpoint["fc_out.weight"][:self.n_classes]
            checkpoint["fc_out.bias"] = checkpoint["fc_out.bias"][:self.n_classes]

            # Load the pre-trained model into the model.
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(self.device)

        # Train base model without test-time compute
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        # Train base model
        model = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            n_splits=1 if self.config.dataset in ["instance-recognition"] else (3 if self.config.dataset in ["part"] else 5),
            n_runs=30,
            num_epochs=self.config.epochs,
            patience=self.config.early_stopping,
            is_augmented=self.config.data_augmentation,
        )

        return model

    def _create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create a new transformer model instance."""
        if self.model_type == "transformer":
            model = Transformer(
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                hidden_dim=self.config.hidden_dimension,
                dropout=self.config.dropout,
            )
        elif self.model_type == "lstm":
            model = LSTM(
                input_size=input_dim,
                hidden_size=self.config.hidden_dimension,
                num_layers=self.config.num_layers,
                output_size=output_dim,
                dropout=self.config.dropout,
            )
        elif self.model_type == "cnn":
            model = CNN(
                input_size=input_dim,
                num_classes=output_dim,
                dropout=self.config.dropout,
            )
        elif self.model_type == "rcnn":
            model = RCNN(
                input_size=input_dim,
                num_classes=output_dim,
                dropout=self.config.dropout,
            )
        elif self.model_type == "mamba":
            model = Mamba(
                d_model=input_dim,
                d_state=self.config.hidden_dimension,
                d_conv=4,
                expand=2,
                depth=self.config.num_layers,
                n_classes=output_dim,
            )
        elif self.model_type == "kan":
            model = KAN(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=self.config.hidden_dimension,
                num_inner_functions=10,
                dropout_rate=self.config.dropout,
                num_layers=self.config.num_layers,
            )
        elif self.model_type == "vae":
            model = VAE(
                input_size=input_dim,
                latent_dim=self.config.hidden_dimension,
                num_classes=output_dim,
                dropout=self.config.dropout,
            )
        elif self.model_type == "moe":
            model = MOE(
                input_dim=input_dim,
                output_dim=output_dim,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                hidden_dim=self.config.hidden_dimension,
                num_experts=4,
                k=2,
            )
        elif self.model_type == "dense":
            model = Dense(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=self.config.dropout,
            )
        elif self.model_type == "ode":
            model = ODE(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=self.config.dropout,
            )
        elif self.model_type == "rwkv":
            model = RWKV(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=self.config.hidden_dimension,
                dropout=self.config.dropout,
            )
        elif self.model_type == "tcn":
            model = TCN(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=self.config.dropout,
            )
        elif self.model_type == "wavenet":
            model = WaveNet(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=self.config.dropout,
            )
        elif self.model_type == "ensemble":
            model = Ensemble(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dimension,
                output_dim=output_dim,
                dropout=self.config.dropout,
            )
        elif self.model_type == "diffusion":
            model = Diffusion(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=self.config.hidden_dimension,
                time_dim=64,
                num_timesteps=1000,
            )
        else:
            raise ValueError(f"Invalid model: {self.model_type}")

        self.logger.info(f"Created model: {model}")
        return model

    def _calculate_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """Calculate balanced class weights."""
        class_counts = {}
        total_samples = 0

        for _, labels in train_loader:
            for label in labels:
                class_label = label.argmax().item()
                class_counts[class_label] = class_counts.get(class_label, 0) + 1
                total_samples += 1

        weights = [1.0 / class_counts[i] for i in range(len(class_counts))]
        weights_tensor = torch.FloatTensor(weights)
        return weights_tensor / weights_tensor.sum() * len(class_counts)

    def _plot_attention_maps(self, model: Transformer, data) -> None:
        """Plot attention maps for model analysis."""
        raise NotImplementedError
        # i = 10
        # columns = data.axes[1][1:(i+1)].tolist()
        # plot_attention_map("encoder", layer_weights, columns, columns)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="Transformer", description="A transformer for fish species classification."
    )

    # File and dataset settings
    parser.add_argument(
        "-f",
        "--file-path",
        type=str,
        default="transformer_checkpoint",
        help="Filepath for model checkpoints",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="species",
        help="Fish species or part dataset. Defaults to species.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="transformer",
        help="Model type to use. Defaults to transformer.",
    )
    parser.add_argument("-r", "--run", type=int, default=0)
    parser.add_argument("-o", "--output", type=str, default="logs/results")

    # Preprocessing flags
    parser.add_argument(
        "-da",
        "--data-augmentation",
        action="store_true",
        help="Enable data augmentation",
        default=False,
    )
    parser.add_argument(
        "-msm",
        "--masked-spectra-modelling",
        action="store_true",
        help="Enable masked spectra modelling",
    )
    parser.add_argument(
        "-nsp",
        "--next-spectra-prediction",
        action="store_true",
        help="Enable next spectra prediction",
    )

    # Regularization settings
    parser.add_argument(
        "-es",
        "--early-stopping",
        type=int,
        default=10,
        help="Early stopping patience. Defaults to 10.",
    )
    parser.add_argument(
        "-do",
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability. Defaults to 0.2.",
    )
    parser.add_argument(
        "-ls",
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing alpha value. Defaults to 0.1.",
    )

    # Training hyperparameters
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs. Defaults to 100.",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate. Defaults to 1e-5.",
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=128, help="Batch size. Defaults to 64."
    )
    parser.add_argument(
        "-hd",
        "--hidden-dimension",
        type=int,
        default=128,
        help="Hidden dimension size. Defaults to 128.",
    )
    parser.add_argument(
        "-l",
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers. Defaults to 4.",
    )
    parser.add_argument(
        "-nh",
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads. Defaults to 4.",
    )

    # Data augmentation parameters
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=5,
        help="Number of augmentations per sample. Defaults to 5.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.1,
        help="Level of noise for augmentation. Defaults to 0.1.",
    )
    parser.add_argument(
        "--shift-enabled", action="store_true", help="Enable shift augmentation"
    )
    parser.add_argument(
        "--scale-enabled", action="store_true", help="Enable scale augmentation"
    )

    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        # Parse arguments and create config
        args = parse_arguments()
        config = TrainingConfig.from_args(args)

        # Initialize trainer
        trainer = ModelTrainer(config)
        trainer.logger.info("Starting training pipeline")

        print(f"config.data_augmentation: {config.data_augmentation}")

        # Create augmentation configuration
        aug_config = AugmentationConfig(
            enabled=config.data_augmentation,
            num_augmentations=config.num_augmentations,
            noise_enabled=True,  # Always enable noise for augmentation
            shift_enabled=config.shift_enabled,
            scale_enabled=config.scale_enabled,
            noise_level=config.noise_level,
        )

        # Create main training data module
        data_module = create_data_module(
            dataset_name=config.dataset,
            batch_size=config.batch_size,
            augmentation_enabled=config.data_augmentation,
            is_pre_train=False,
            **{
                "num_augmentations": config.num_augmentations,
                "noise_level": config.noise_level,
                "shift_enabled": config.shift_enabled,
                "scale_enabled": config.scale_enabled,
            },
        )

        # # Setup main data module to get loaders
        # train_loader, data = data_module.setup()
        # trainer.logger.info(f"Main training dataset size: {len(train_loader.dataset)} samples")

        # Run pre-training if enabled
        pre_trained_model = None
        if config.masked_spectra_modelling or config.next_spectra_prediction:
            # Create pre-training data module
            pretrain_data_module = create_data_module(
                dataset_name=config.dataset,
                batch_size=config.batch_size,
                augmentation_enabled=config.data_augmentation,
                is_pre_train=True,
                **{
                    "num_augmentations": config.num_augmentations,
                    "noise_level": config.noise_level,
                    "shift_enabled": config.shift_enabled,
                    "scale_enabled": config.scale_enabled,
                },
            )

            # Setup pre-training data module
            pretrain_loader, _ = pretrain_data_module.setup()
            trainer.logger.info(
                f"Pre-training dataset size: {len(pretrain_loader.dataset)} samples"
            )

            # Update trainer's data module for pre-training
            trainer.data_module = pretrain_data_module
            pre_trained_model = trainer.pre_train()

            if pre_trained_model is not None:
                trainer.logger.info("Pre-training completed successfully")

        # Update trainer's data module for main training
        trainer.data_module = data_module

        # Run main training
        model = trainer.train(pre_trained_model)

        return model

    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

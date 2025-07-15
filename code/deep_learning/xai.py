from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from models import *  # Import all models
from .train import train_model
from .util import create_data_module

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for neural network models."""

    input_dim: int
    output_dim: int
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    # Optional model-specific parameters
    num_heads: Optional[int] = None  # Transformer
    d_conv: int = 4  # Mamba
    expand: int = 2  # Mamba
    num_inner_functions: int = 10  # KAN
    latent_dim: Optional[int] = None  # VAE


@dataclass
class TrainConfig:
    """Configuration for model training."""

    batch_size: int = 64
    learning_rate: float = 1e-5
    num_epochs: int = 100
    patience: int = 10
    label_smoothing: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExplainerConfig:
    """Configuration for model explanation."""

    num_features: int = 5
    num_samples: int = 100
    output_dir: Path = Path("figures")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42


class ModelFactory:
    """Factory for creating different model architectures."""

    @staticmethod
    def create(model_name: str, config: ModelConfig) -> nn.Module:
        """Create a model instance based on configuration.

        Args:
            model_name (str): Name of the model architecture to create.
            config (ModelConfig): Configuration parameters for the model.

        Returns:
            nn.Module: An instance of the specified model architecture.
        """
        models = {
            "transformer": lambda: Transformer(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads or 4,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            ),
            "lstm": lambda: LSTM(
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                output_size=config.output_dim,
                dropout=config.dropout,
            ),
            "cnn": lambda: CNN(
                input_size=config.input_dim,
                num_classes=config.output_dim,
                dropout=config.dropout,
            ),
            "rcnn": lambda: RCNN(
                input_size=config.input_dim,
                num_classes=config.output_dim,
                dropout=config.dropout,
            ),
            "mamba": lambda: Mamba(
                d_model=config.input_dim,
                d_state=config.hidden_dim,
                d_conv=config.d_conv,
                expand=config.expand,
                depth=config.num_layers,
                n_classes=config.output_dim,
            ),
            "kan": lambda: KAN(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                hidden_dim=config.hidden_dim,
                num_inner_functions=config.num_inner_functions,
                dropout_rate=config.dropout,
                num_layers=config.num_layers,
            ),
            "vae": lambda: VAE(
                input_size=config.input_dim,
                latent_dim=config.latent_dim or config.hidden_dim,
                num_classes=config.output_dim,
                dropout=config.dropout,
            ),
        }

        if model_name not in models:
            raise ValueError(f"Unknown model type: {model_name}")

        return models[model_name]()


class ModelWrapper:
    """Wrapper for neural network models to make them compatible with LIME."""

    def __init__(self, model: nn.Module, device: str) -> Nones:
        """Initialize the model wrapper.

        Args:
            model (nn.Module): The neural network model to wrap.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def normalize_intensities(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize mass spec intensities to range [0,1]

        Args:
            x (np.ndarray): Input array of mass spec intensities, shape (n_samples, n_features).

        Returns:
            np.ndarray: Normalized intensities, same shape as input.
        """
        # Handle edge case where all values are the same
        if np.all(x == x[0]):
            return np.zeros_like(x)

        # Min-max normalization for each spectrum
        x_norm = np.zeros_like(x)
        for i in range(len(x)):
            spectrum = x[i]
            min_val = np.min(spectrum)
            max_val = np.max(spectrum)

            # Avoid division by zero
            if max_val - min_val > 0:
                x_norm[i] = (spectrum - min_val) / (max_val - min_val)
            else:
                x_norm[i] = np.zeros_like(spectrum)

        return x_norm

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get probability predictions from model with normalized intensities.

        Args:
            x (np.ndarray): Input array of mass spec intensities, shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability predictions, shape (n_samples, n_classes).
        """
        try:
            # Normalize intensities to [0,1] range
            x_normalized = self.normalize_intensities(x)

            # Convert to tensor and move to device
            x_tensor = torch.tensor(x_normalized, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                if isinstance(self.model, VAE):
                    _, _, _, logits = self.model(x_tensor)
                else:
                    logits = (
                        self.model(x_tensor, x_tensor)
                        if isinstance(self.model, Transformer)
                        else self.model(x_tensor)
                    )

                return F.softmax(logits, dim=-1).cpu().numpy()

        except Exception as e:
            logger.error(f"Error in predict_proba: {str(e)}")
            raise


class ModelExplainer:
    """Explains predictions of neural network models using LIME."""

    DATASET_LABELS = {
        "species": ["Hoki", "Mackerel"],
        "part": ["Fillet", "Heads", "Livers", "Skins", "Guts", "Gonads", "Frames"],
        "oil": ["50", "25", "10", "05", "01", "0.1", "0"],
        "oil_simple": ["Oil", "No oil"],
        "cross-species": ["Hoki-Mackeral", "Hoki", "Mackerel"],
        "instance-recognition": ["different", "same"],
    }

    def __init__(self, model: nn.Module, config: ExplainerConfig) -> None:
        """Initialize model explainer.

        Args:
            model (nn.Module): The neural network model to explain.
            config (ExplainerConfig): Configuration parameters for the explainer.

        Raises:
            ValueError: If the dataset name is not recognized.
        """
        self.config = config
        self.device = config.device
        self.model_wrapper = ModelWrapper(model, config.device)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def normalize_features(self, features: torch.Tensor) -> np.ndarray:
        """
        Normalize features to [0,1] range before LIME explanation.

        Args:
            features (torch.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            np.ndarray: Normalized features, shape (n_samples, n_features).
        """
        # Convert to numpy and ensure 2D
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        # Apply min-max normalization per spectrum
        normalized = np.zeros_like(features_np, dtype=np.float32)
        for i in range(len(features_np)):
            spectrum = features_np[i]
            min_val = np.min(spectrum)
            max_val = np.max(spectrum)

            if max_val - min_val > 1e-8:  # Avoid division by zero
                normalized[i] = (spectrum - min_val) / (max_val - min_val)
            else:
                normalized[i] = np.zeros_like(spectrum)

        return normalized

    def setup_explainer(
        self,
        dataset_name: str,
        features: torch.Tensor,
        labels: torch.Tensor,
        feature_names: List[str],
    ) -> LimeTabularExplainer:
        """Set up LIME explainer for the model.

        Args:
            dataset_name (str): Name of the dataset to explain.
            features (torch.Tensor): Input features tensor of shape (n_samples, n_features).
            labels (torch.Tensor): Corresponding labels tensor of shape (n_samples,).
            feature_names (List[str]): List of feature names.

        Returns:
            LimeTabularExplainer: Configured LIME explainer instance.
        """
        if dataset_name not in self.DATASET_LABELS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        try:
            # Use normalized features directly
            normalized_features = self.normalize_features(features)

            return LimeTabularExplainer(
                training_data=normalized_features,
                training_labels=labels.cpu().numpy(),
                feature_names=feature_names,
                class_names=self.DATASET_LABELS[dataset_name],
                discretize_continuous=True,
                random_state=self.config.random_seed,
            )
        except Exception as e:
            logger.error(f"Failed to setup LIME explainer: {e}")
            raise

    def find_instance(
        self, features: torch.Tensor, labels: torch.Tensor, target: List[float]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Find an instance with the target label.

        Args:
            features (torch.Tensor): Input features tensor of shape (n_samples, n_features).
            labels (torch.Tensor): Corresponding labels tensor of shape (n_samples,).
            target (List[float]): Target label to find, e.g., [0, 0, 0, 0, 0, 0, 1].

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: The feature and label of the found instance,
            or (None, None) if no instance matches the target label.
        """
        try:
            target_tensor = torch.tensor(target)
            for feat, label in zip(features, labels):
                if torch.equal(label, target_tensor):
                    return feat, label
            logger.warning(f"No instance found with label {target}")
            return None, None
        except Exception as e:
            logger.error(f"Error finding instance: {e}")
            raise

    def explain(
        self, instance: torch.Tensor, explainer: LimeTabularExplainer, output_path: Path
    ) -> None:
        """Generate and save LIME explanation with improved readability.

        Args:
            instance (torch.Tensor): Input instance tensor of shape (n_features,).
            explainer (LimeTabularExplainer): Configured LIME explainer instance.
            output_path (Path): Path to save the explanation figure.

        Raises:
            Exception: If explanation generation fails.
        """
        try:
            # Generate LIME explanation
            normalized_instance = self.normalize_features(instance)
            explanation = explainer.explain_instance(
                normalized_instance.flatten(),
                self.model_wrapper.predict_proba,
                num_features=self.config.num_features,
                num_samples=self.config.num_samples,
            )

            # First get the figure from LIME
            fig = explanation.as_pyplot_figure()

            # Now modify the existing figure
            ax = plt.gca()

            # Adjust figure size
            fig.set_size_inches(8, 6)

            # Increase y-axis tick label size
            ax.tick_params(axis="y", labelsize=12)

            # Make bars thinner
            for patch in ax.patches:
                current_height = patch.get_height()
                new_height = current_height * 0.7
                current_y = patch.get_y()
                adjustment = (current_height - new_height) / 2
                patch.set_height(new_height)
                patch.set_y(current_y + adjustment)

            # Style customization
            ax.set_facecolor("#E6E6E6")
            fig.patch.set_facecolor("white")
            ax.grid(
                True, color="white", linestyle="-", linewidth=1.0, alpha=1.0, zorder=0
            )
            ax.set_axisbelow(True)

            # Adjust layout and save
            plt.tight_layout()
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Saved explanation to {output_path}")

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            raise


def explain_predictions(
    dataset_name: str,
    model_name: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    explainer_config: ExplainerConfig,
    instance_name: str,
    target_label: List[float],
) -> None:
    """Generate explanations for model predictions.

    Args:
        dataset_name (str): Name of the dataset to explain.
        model_name (str): Name of the model architecture to use.
        model_config (ModelConfig): Configuration parameters for the model.
        train_config (TrainConfig): Configuration parameters for training.
        explainer_config (ExplainerConfig): Configuration parameters for the explainer.
        instance_name (str): Name of the specific instance to explain.
        target_label (List[float]): Target label to find and explain.

    Raises:
        Exception: If the explanation pipeline fails at any step.
    """
    try:
        # Create and train model
        model = ModelFactory.create(model_name, model_config)
        model.to(train_config.device)

        # Create data module and load dataset
        data_module = create_data_module(
            file_path="/home/woodj/Desktop/fishy-business/data/REIMS.xlsx",  # Assuming this is needed for main training
            # file_path = "/vol/ecrg-solar/woodj4/fishy-business/data/REIMS.xlsx" # Example server path
            dataset_name=dataset_name,  # Use your desired dataset
            batch_size=32,
        )

        # Setup the data loader
        data_module.setup()
        train_loader = data_module.get_train_dataloader()
        data = data_module.get_train_dataframe()

        # Train model
        criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)

        model, _ = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=train_config.num_epochs,
            patience=train_config.patience,
            n_runs=1,
            n_splits=1,
        )

        # Setup explainer
        explainer = ModelExplainer(model, explainer_config)

        # Get feature names and first batch
        feature_names = [f"{float(x):.4f}" for x in data.axes[1].tolist()[1:]]
        features, labels = next(iter(train_loader))

        # Generate explanation
        lime_explainer = explainer.setup_explainer(
            dataset_name, features, labels, feature_names
        )
        instance, label = explainer.find_instance(features, labels, target_label)

        if instance is not None:
            output_path = (
                explainer_config.output_dir
                / dataset_name
                / f"lime_{model_name}_{dataset_name}_{instance_name}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            explainer.explain(instance, lime_explainer, output_path)
        else:
            logger.warning(f"No instance found with label {target_label}")

    except Exception as e:
        logger.error(f"Explanation pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    """Main entry point for testing the explanation functionality."""
    # Example usage
    model_config = ModelConfig(input_dim=2080, output_dim=7, num_heads=4)

    train_config = TrainConfig()
    explainer_config = ExplainerConfig()

    # "species": ["Hoki", "Mackerel"],
    # "part": ["Fillet", "Heads", "Livers", "Skins", "Guts", "Gonads", "Frames"],

    explain_predictions(
        dataset_name="part",
        model_name="transformer",
        model_config=model_config,
        train_config=train_config,
        explainer_config=explainer_config,
        instance_name="frames",
        target_label=[0, 0, 0, 0, 0, 0, 1],
    )

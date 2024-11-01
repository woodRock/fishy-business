from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from transformer import Transformer
from train import train_model
from util import preprocess_dataset

# Configure logging
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExplanationConfig:
    """Configuration for LIME explanations."""

    num_features: int = 5
    num_samples: int = 100
    output_dir: str = "figures"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42


def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class ModelWrapper:
    """Wrapper for transformer model to make it compatible with LIME."""

    def __init__(self, model: nn.Module, device: str):
        """Initialize model wrapper.

        Args:
            model: PyTorch model to wrap
            device: Device to run model on
        """
        self.model = model
        self.device = device
        self.model.eval()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get probability predictions from model.

        Args:
            x: Input features

        Returns:
            Probability predictions
        """
        try:
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.model(x, x)  # Using same input for src and tgt
                probs = F.softmax(logits, dim=-1).cpu().numpy()
            return probs
        except Exception as e:
            logger.error(f"Error in predict_proba: {str(e)}")
            raise


class TransformerExplainer:
    """Handles LIME explanations for transformer models."""

    LABELS_PER_DATASET = {
        "species": ["Hoki", "Mackerel"],
        "part": ["Fillet", "Heads", "Livers", "Skins", "Guts", "Gonads", "Frames"],
        "oil": ["50", "25", "10", "05", "01", "0.1", "0"],
        "oil_simple": ["Oil", "No oil"],
        "cross-species": ["Hoki-Mackeral", "Hoki", "Mackerel"],
        "instance-recognition": ["different", "same"],
    }

    def __init__(
        self, dataset_name: str, model: Transformer, config: ExplanationConfig
    ):
        """Initialize explainer.

        Args:
            dataset_name: Name of dataset
            model: Transformer model to explain
            config: Explanation configuration
        """
        if dataset_name not in self.LABELS_PER_DATASET:
            raise ValueError(f"Invalid dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.config = config
        self.wrapped_model = ModelWrapper(model, config.device)
        self.class_names = self.LABELS_PER_DATASET[dataset_name]

        # Setup output directory
        self.output_dir = Path(config.output_dir) / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_explainer(
        self, features: torch.Tensor, labels: torch.Tensor, feature_names: List[str]
    ) -> LimeTabularExplainer:
        """Setup LIME explainer with data.

        Args:
            features: Training features
            labels: Training labels
            feature_names: Names of features

        Returns:
            Configured LIME explainer
        """
        try:
            # Standardize data
            scaler = StandardScaler()
            standardized_features = scaler.fit_transform(features.cpu().numpy())

            return LimeTabularExplainer(
                training_data=standardized_features,
                training_labels=labels.cpu().numpy(),
                feature_names=feature_names,
                class_names=self.class_names,
                discretize_continuous=True,
                random_state=self.config.random_seed,
            )
        except Exception as e:
            logger.error(f"Error setting up explainer: {str(e)}")
            raise

    def find_instance_by_label(
        self, features: torch.Tensor, labels: torch.Tensor, target_label: List[float]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Find first instance with specific label.

        Args:
            features: All features
            labels: All labels
            target_label: Label to find

        Returns:
            Tuple of (instance features, instance label) or (None, None)
        """
        try:
            target_tensor = torch.tensor(target_label)
            for f, l in zip(features, labels):
                if torch.equal(l, target_tensor):
                    return f, l
            logger.warning(f"No instance found with label {target_label}")
            return None, None
        except Exception as e:
            logger.error(f"Error finding instance: {str(e)}")
            raise

    def generate_explanation(
        self, instance: torch.Tensor, explainer: LimeTabularExplainer, output_file: str
    ) -> None:
        """Generate and save LIME explanation.

        Args:
            instance: Instance to explain
            explainer: LIME explainer
            output_file: Path to save visualization
        """
        try:
            # Generate explanation
            explanation = explainer.explain_instance(
                instance.cpu().numpy(),
                self.wrapped_model.predict_proba,
                num_features=self.config.num_features,
                num_samples=self.config.num_samples,
            )

            # Create visualization
            fig = explanation.as_pyplot_figure()
            fig.set_size_inches(10, 8)
            plt.tight_layout()

            # Save figure
            output_path = self.output_dir / output_file
            fig.savefig(output_path)
            plt.close(fig)

            logger.info(f"Saved explanation to {output_path}")
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise


def explain_transformer_predictions(
    dataset_name: str = "cross-species",
    instance_name: str = "hoki-mackerel",
    model_name: str = "transformer",
    target_label = [0, 0, 1],  # Example for Mackerel
    input_dim: int = 2080,
    output_dim: int = 3,
    num_layers: int = 4,
    num_heads: int = 4,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    num_epochs: int = 100,
    patience: int = 10,
) -> None:
    """Generate LIME explanations for transformer predictions."""
    try:
        # Setup logging
        setup_logging()
        logger.info(f"Starting explanation generation for dataset: {dataset_name}")

        # Initialize model
        model = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Setup device and move model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Using device: {device}")

        # Load data
        train_loader, data = preprocess_dataset(
            dataset=dataset_name,
            batch_size=batch_size,
            is_data_augmentation=False,
            is_pre_train=False,
        )
        logger.info(f"Loaded dataset with {len(train_loader.dataset)} samples")

        # Train model
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        model = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            patience=patience,
        )
        logger.info("Model training completed")

        # Setup explainer
        config = ExplanationConfig()
        explainer = TransformerExplainer(dataset_name, model, config)

        # Get feature names
        feature_names = [f"{float(x):.4f}" for x in data.axes[1].tolist()[1:]]

        # Get first batch of data
        features, labels = next(iter(train_loader))
        logger.info("Setting up LIME explainer")

        # Setup LIME explainer
        lime_explainer = explainer.setup_explainer(features, labels, feature_names)

        # Find specific instance (e.g., Mackerel in cross-species)
        instance, label = explainer.find_instance_by_label(
            features, labels, target_label
        )

        if instance is not None:
            # Generate explanation
            explainer.generate_explanation(
                instance,
                lime_explainer,
                f"lime_{model_name}_{dataset_name}_{instance_name}.png",
            )
            logger.info("Explanation generation completed successfully")
        else:
            logger.warning(f"No instance found with label {target_label}")

    except Exception as e:
        logger.error(f"Error in explanation pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    explain_transformer_predictions(
        dataset_name="cross-species",
        instance_name="hoki-mackerel",
        model_name="transformer",
        target_label=[0,0,1]
    )

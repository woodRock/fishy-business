from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Type
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from lstm import LSTM
from transformer import Transformer
from cnn import CNN
from rcnn import RCNN
from mamba import Mamba
from kan import KAN
from vae import VAE
from train import train_model
from util import preprocess_dataset

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
        """Create a model instance based on configuration."""
        models = {
            "transformer": lambda: Transformer(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads or 4,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout
            ),
            "lstm": lambda: LSTM(
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                output_size=config.output_dim,
                dropout=config.dropout
            ),
            "cnn": lambda: CNN(
                input_size=config.input_dim,
                num_classes=config.output_dim,
                dropout=config.dropout
            ),
            "rcnn": lambda: RCNN(
                input_size=config.input_dim,
                num_classes=config.output_dim,
                dropout=config.dropout
            ),
            "mamba": lambda: Mamba(
                d_model=config.input_dim,
                d_state=config.hidden_dim,
                d_conv=config.d_conv,
                expand=config.expand,
                depth=config.num_layers,
                n_classes=config.output_dim
            ),
            "kan": lambda: KAN(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                hidden_dim=config.hidden_dim,
                num_inner_functions=config.num_inner_functions,
                dropout_rate=config.dropout,
                num_layers=config.num_layers
            ),
            "vae": lambda: VAE(
                input_size=config.input_dim,
                latent_dim=config.latent_dim or config.hidden_dim,
                num_classes=config.output_dim,
                dropout=config.dropout
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model type: {model_name}")
            
        return models[model_name]()

class ModelWrapper:
    """Wrapper for neural network models to make them compatible with LIME."""

    def __init__(self, model: nn.Module, device: str):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.scaler = StandardScaler()

    def normalize_intensities(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize mass spec intensities to range [0,1]
        
        Parameters:
        -----------
        x : np.ndarray
            Input spectral data
            
        Returns:
        --------
        np.ndarray
            Normalized spectral data with intensities between 0 and 1
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
        """Get probability predictions from model with normalized intensities."""
        try:
            # Normalize intensities to [0,1] range
            x_normalized = self.normalize_intensities(x)
            
            # Convert to tensor and move to device
            x_tensor = torch.tensor(x_normalized, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                if isinstance(self.model, VAE):
                    _, _, _, logits = self.model(x_tensor)
                else:
                    logits = self.model(x_tensor, x_tensor) if isinstance(self.model, Transformer) else self.model(x_tensor)
                
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
        "instance-recognition": ["different", "same"]
    }

    def __init__(self, model: nn.Module, config: ExplainerConfig):
        """Initialize model explainer."""
        self.config = config
        self.device = config.device
        self.model_wrapper = ModelWrapper(model, config.device)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_explainer(self, dataset_name: str, features: torch.Tensor, 
                       labels: torch.Tensor, feature_names: List[str]) -> LimeTabularExplainer:
        """Set up LIME explainer for the model."""
        if dataset_name not in self.DATASET_LABELS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        try:
            scaler = StandardScaler()
            standardized_features = scaler.fit_transform(features.cpu().numpy())
            
            return LimeTabularExplainer(
                training_data=standardized_features,
                training_labels=labels.cpu().numpy(),
                feature_names=feature_names,
                class_names=self.DATASET_LABELS[dataset_name],
                discretize_continuous=True,
                random_state=self.config.random_seed
            )
        except Exception as e:
            logger.error(f"Failed to setup LIME explainer: {e}")
            raise

    def find_instance(self, features: torch.Tensor, labels: torch.Tensor, 
                     target: List[float]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Find an instance with the target label."""
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

    def explain(self, instance: torch.Tensor, explainer: LimeTabularExplainer, 
                output_path: Path) -> None:
        """Generate and save LIME explanation."""
        try:
            explanation = explainer.explain_instance(
                instance.cpu().numpy(),
                self.model_wrapper.predict_proba,
                num_features=self.config.num_features,
                num_samples=self.config.num_samples
            )

            fig = explanation.as_pyplot_figure()
            fig.set_size_inches(10, 8)
            plt.tight_layout()
            
            fig.savefig(output_path)
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
    target_label: List[float]
) -> None:
    """Generate explanations for model predictions."""
    try:
        # Create and train model
        model = ModelFactory.create(model_name, model_config)
        model.to(train_config.device)
        
        # Load dataset
        train_loader, data = preprocess_dataset(
            dataset=dataset_name,
            batch_size=train_config.batch_size,
            is_pre_train=False
        )
        
        # Train model
        criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
        model = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=train_config.num_epochs,
            patience=train_config.patience
        )
        
        # Setup explainer
        explainer = ModelExplainer(model, explainer_config)
        
        # Get feature names and first batch
        feature_names = [f"{float(x):.4f}" for x in data.axes[1].tolist()[1:]]
        features, labels = next(iter(train_loader))
        
        # Generate explanation
        lime_explainer = explainer.setup_explainer(dataset_name, features, labels, feature_names)
        instance, label = explainer.find_instance(features, labels, target_label)
        
        if instance is not None:
            output_path = explainer_config.output_dir / dataset_name / f"lime_{model_name}_{dataset_name}_{instance_name}.png"
            explainer.explain(instance, lime_explainer, output_path)
        else:
            logger.warning(f"No instance found with label {target_label}")
            
    except Exception as e:
        logger.error(f"Explanation pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Example usage
    model_config = ModelConfig(
        input_dim=2080,
        output_dim=7,
        num_heads=4
    )
    
    train_config = TrainConfig()
    explainer_config = ExplainerConfig()
    
    explain_predictions(
        dataset_name="part",
        model_name="transformer",
        model_config=model_config,
        train_config=train_config,
        explainer_config=explainer_config,
        instance_name="gonads",
        target_label=[0, 0, 0, 0, 0, 1, 0]
    )
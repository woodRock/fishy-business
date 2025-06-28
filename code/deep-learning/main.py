import argparse
import logging
import time
import traceback
from dataclasses import dataclass, fields as dataclass_fields # For TrainingConfig.from_args (optional conciseness)
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from sklearn.metrics import ( # These are used in train.py, ensure not needed here directly if already there
#     balanced_accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score
# )
# from collections import defaultdict # Not directly used in this snippet
# from torch.utils.data import DataLoader, Subset, random_split # Subset/random_split not directly used
# from sklearn.model_selection import StratifiedKFold # Used in train.py

import numpy as np

from models import * # Import all models from models/__init__.py
from .pre_training import PreTrainer, PreTrainingConfig
from .train import train_model # Your refactored train_model script
from .util import preprocess_data_pipeline, create_data_module, AugmentationConfig


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    file_path: str = "transformer_checkpoint.pth"
    model_type: str = "transformer"
    dataset: str = "species"
    run: int = 0
    output: str = "logs/results"
    data_augmentation: bool = False
    masked_spectra_modelling: bool = False
    next_spectra_prediction: bool = False
    next_peak_prediction: bool = False
    spectrum_denoising_autoencoding: bool = False # Typo: autoencoding
    peak_parameter_regression: bool = False
    spectrum_segment_reordering: bool = False
    contrastive_transformation_invariance_learning: bool = False
    early_stopping: int = 10
    dropout: float = 0.2
    label_smoothing: float = 0.1
    epochs: int = 100
    learning_rate: float = 1e-5
    batch_size: int = 64
    hidden_dimension: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_augmentations: int = 5
    noise_level: float = 0.1
    shift_enabled: bool = False
    scale_enabled: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create configuration from command line arguments."""
        # Current explicit mapping is clear and robust to naming differences.
        # For extreme conciseness if field names match argparse dest:
        # return cls(**{f.name: getattr(args, f.name) for f in dataclass_fields(cls) if hasattr(args, f.name)})
        return cls(
            file_path=args.file_path, dataset=args.dataset, model_type=args.model,
            run=args.run, output=args.output, data_augmentation=args.data_augmentation,
            num_augmentations=args.num_augmentations, noise_level=args.noise_level,
            shift_enabled=args.shift_enabled, scale_enabled=args.scale_enabled,
            masked_spectra_modelling=args.masked_spectra_modelling,
            next_spectra_prediction=args.next_spectra_prediction,
            next_peak_prediction=args.next_peak_prediction,
            spectrum_denoising_autoencoding=args.spectrum_denoising_autoencoding,
            peak_parameter_regression=args.peak_parameter_regression,
            spectrum_segment_reordering=args.spectrum_segment_reordering,
            contrastive_transformation_invariance_learning=args.contrastive_transformation_invariance_learning,
            early_stopping=args.early_stopping, dropout=args.dropout,
            label_smoothing=args.label_smoothing, epochs=args.epochs,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            hidden_dimension=args.hidden_dimension, num_layers=args.num_layers,
            num_heads=args.num_heads,
        )

class ModelTrainer:
    N_CLASSES_PER_DATASET = {
        "species": 2, "part": 7, "oil": 7, "cross-species": 3,
        "cross-species-hard": 15, "instance-recognition": 2,
        "instance-recognition-hard": 24,
    }
    # (config_flag_name, output_dim_calculator_fn, pre_trainer_method_name, requires_val_loader, additional_method_kwargs)
    PRETRAIN_TASK_DEFINITIONS: List[Tuple[str, Callable[["ModelTrainer"], int], str, bool, Dict[str, Any]]] = [
        ("masked_spectra_modelling", lambda self: self.n_features, "pre_train_masked_spectra", False, {}),
        ("next_spectra_prediction", lambda self: 2, "pre_train_next_spectra", True, {}),
        ("next_peak_prediction", lambda self: self.n_features, "pre_train_peak_prediction", False, {"peak_threshold": 0.1, "window_size": 5}),
        ("spectrum_denoising_autoencoding", lambda self: self.n_features, "pre_train_denoising_autoencoder", False, {}), # Typo: autoencoding
        ("peak_parameter_regression", lambda self: self.n_features, "pre_train_peak_parameter_regression", False, {}),
        ("spectrum_segment_reordering", lambda self: 4*4, "pre_train_spectrum_segment_reordering", False, {"num_segments": 4}),
        ("contrastive_transformation_invariance_learning", lambda self: 128, "pre_train_contrastive_invariance", False, {"embedding_dim": 128}),
    ]

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model_type = config.model_type
        self.data_module = None # Will be set before pre_train or train

        if config.dataset not in self.N_CLASSES_PER_DATASET:
            raise ValueError(f"Invalid dataset: {config.dataset}")
        self.n_classes = self.N_CLASSES_PER_DATASET[config.dataset]
        self.n_features = 2080

    def _setup_logging(self) -> logging.Logger:
        log_file = Path(self.config.output).parent / f"{Path(self.config.output).name}_{self.config.run}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_file, level=logging.INFO, filemode="w",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        # Add console handler to see logs in terminal as well
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger = logging.getLogger(__name__)
        if not logger.handlers: # Avoid adding handlers multiple times if already configured
            logger.addHandler(console_handler)
            # If basicConfig added a FileHandler, it might be duplicated if not careful.
            # For simplicity, assuming basicConfig is the primary file logger config.
        logger.propagate = False # Prevent root logger from handling these messages again
        return logger

    def _handle_weight_chaining(self, current_model: nn.Module, prev_model: Optional[nn.Module]):
        if not prev_model:
            return
        self.logger.info(f"Attempting to load weights from previous pre-trained model for {self.model_type}")
        try:
            prev_state_dict = prev_model.state_dict()
            current_model_dict = current_model.state_dict()
            
            # Filter keys: only load if key exists in current model and shape matches
            # More sophisticated filtering for final layers might be needed if output_dim changes drastically
            load_state_dict = {
                k: v for k, v in prev_state_dict.items()
                if k in current_model_dict and v.shape == current_model_dict[k].shape
            }
            missing_keys, unexpected_keys = current_model.load_state_dict(load_state_dict, strict=False)
            if missing_keys: self.logger.warning(f"Chaining: Missing keys: {missing_keys}")
            if unexpected_keys: self.logger.warning(f"Chaining: Unexpected keys: {unexpected_keys}")
            self.logger.info("Weight chaining: successfully loaded compatible weights.")
        except Exception as e:
            self.logger.warning(f"Weight chaining: Could not load weights: {e}. Model will train from scratch or partially loaded state.")

    def _execute_pretrain_task(self, task_name: str, model_to_chain_from: Optional[nn.Module],
                               train_loader: DataLoader, pre_train_cfg: PreTrainingConfig,
                               output_dim: int, pre_trainer_method: str,
                               requires_val: bool, method_kwargs: Dict[str, Any]) -> Optional[nn.Module]:
        self.logger.info(f"Starting pre-training task: {task_name}")
        current_model = self._create_model(input_dim=self.n_features, output_dim=output_dim).to(self.device)
        self._handle_weight_chaining(current_model, model_to_chain_from)

        pre_trainer = PreTrainer(
            model=current_model, config=pre_train_cfg,
            optimizer=torch.optim.AdamW(current_model.parameters(), lr=self.config.learning_rate)
        )
        
        call_args = [train_loader]
        if requires_val:
            # Assuming data_module (set to pretrain_data_module) can provide a val_loader
            # This part might need adjustment based on your DataModule's API
            _, val_loader = self.data_module.setup() # Or a more specific val_loader method
            if val_loader is None and hasattr(self.data_module, 'setup_val'): # Hypothetical
                 _, val_loader = self.data_module.setup_val()

            if val_loader is None: self.logger.warning(f"Validation loader for {task_name} not found, passing None.")
            call_args.append(val_loader)
        
        start_time = time.time()
        trained_model = getattr(pre_trainer, pre_trainer_method)(*call_args, **method_kwargs)
        self.logger.info(f"{task_name} training time: {time.time() - start_time:.2f}s")
        return trained_model

    def pre_train(self) -> Optional[nn.Module]:
        self.logger.info("Evaluating pre-training phase")
        enabled_task_flags = [task_def[0] for task_def in self.PRETRAIN_TASK_DEFINITIONS if getattr(self.config, task_def[0], False)]
        if not enabled_task_flags:
            self.logger.info("No pre-training tasks enabled.")
            return None

        self.logger.info(f"Enabled pre-training tasks: {', '.join(enabled_task_flags)}")
        if self.data_module is None: # Should be set by caller
            self.logger.error("Pre-training DataModule not set in ModelTrainer.")
            return None
        train_loader, _ = self.data_module.setup()

        pre_train_cfg = PreTrainingConfig(
            num_epochs=self.config.epochs, file_path=self.config.file_path,
            device=self.device, n_features=self.n_features
        )
        model_after_last_task: Optional[nn.Module] = None
        for flag, out_dim_fn, method, req_val, kwargs in self.PRETRAIN_TASK_DEFINITIONS:
            if getattr(self.config, flag, False):
                output_dim = out_dim_fn(self)
                model_after_last_task = self._execute_pretrain_task(
                    flag, model_after_last_task, train_loader, pre_train_cfg,
                    output_dim, method, req_val, kwargs
                )
        self.logger.info(f"Pre-training completed. Final model from task: {'Yes' if model_after_last_task else 'No'}")
        return model_after_last_task

    def _calculate_metrics(self, true_labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        metrics = {
            'accuracy': balanced_accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='macro', zero_division=0),
            'recall': recall_score(true_labels, predictions, average='macro', zero_division=0),
            'f1': f1_score(true_labels, predictions, average='macro', zero_division=0)
        }
        self.logger.debug("Per-class metrics:")
        for i in range(self.n_classes):
            class_mask = true_labels == i
            if np.sum(class_mask) == 0: continue # Skip if class not in true_labels for this batch
            class_preds = predictions[class_mask]
            # Log individual class scores if needed, for brevity, macro scores are primary
            # self.logger.debug(f"Class {i}: True={np.sum(class_mask)}, Pred={np.sum(predictions==i)}")
        return metrics

    def _evaluate_fold(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        all_outputs, all_true_labels = [], []
        with torch.no_grad():
            for x, y_labels in val_loader: # Assuming y_labels are class indices or one-hot
                x = x.to(self.device)
                all_outputs.append(model(x))
                all_true_labels.append(y_labels) # Keep on CPU for now, or move to device if y was used there

        outputs_tensor = torch.cat(all_outputs, dim=0)
        labels_tensor = torch.cat(all_true_labels, dim=0) # Assuming labels are batch-consistent

        # Convert labels to class indices if they are one-hot
        if labels_tensor.dim() > 1 and labels_tensor.shape[-1] > 1:
            true_labels_np = labels_tensor.argmax(dim=-1).cpu().numpy()
        else: # Assume already class indices
            true_labels_np = labels_tensor.cpu().numpy()
        
        pred_np = outputs_tensor.argmax(dim=-1).cpu().numpy()
        
        # Confidence stats
        probs = torch.softmax(outputs_tensor, dim=-1).max(dim=-1)[0]
        self.logger.info(f"Val Confidence - Mean: {probs.mean():.3f}, Std: {probs.std():.3f}")
        
        return self._calculate_metrics(true_labels_np, pred_np)

    def _get_n_splits_for_finetune(self) -> int:
        if self.config.dataset == "instance-recognition": return 1
        if self.config.dataset == "part": return 3
        return 5 # Default for others like "species", "oil"

    def train(self, pre_trained_model: Optional[nn.Module] = None) -> nn.Module:
        self.logger.info("Starting main fine-tuning phase")
        if self.data_module is None: # Should be set by caller
            self.logger.error("Fine-tuning DataModule not set in ModelTrainer.")
            return self._create_model(self.n_features, self.n_classes) # Return a fresh model
            
        self.data_module.setup() # Load and process data
    
        train_loader = self.data_module.get_train_dataloader()

        model_to_finetune = self._create_model(self.n_features, self.n_classes).to(self.device)
        
        if pre_trained_model:
            self.logger.info("Transferring pre-trained weights for fine-tuning")
            # Adapt final layer of pre_trained_model's state_dict before loading
            # This logic is highly dependent on your model's output layer name (e.g., "fc_out", "fc")
            # And assumes pre_trained_model is on CPU or its state_dict is moved to CPU.
            checkpoint = pre_trained_model.state_dict() 
            final_layer_name = "fc_out" # Change if your model uses a different name
            
            if f"{final_layer_name}.weight" in checkpoint and checkpoint[f"{final_layer_name}.weight"].shape[0] != self.n_classes:
                self.logger.info(f"Adapting final layer '{final_layer_name}' from pre-trained model for {self.n_classes} classes.")
                # Simple truncation/padding or re-initialization might be needed.
                # This is a placeholder for robust adaptation.
                # For example, if Transformer always has fc_out:
                # model_to_finetune.fc_out = nn.Linear(model_to_finetune.fc_out.in_features, self.n_classes).to(self.device)
                # Then load state_dict with strict=False, or adapt checkpoint dict carefully.
                
                # A common approach: load all but the final layer, then re-init final layer of model_to_finetune
                fc_weight_key = f"{final_layer_name}.weight"
                fc_bias_key = f"{final_layer_name}.bias"
                if fc_weight_key in checkpoint: del checkpoint[fc_weight_key]
                if fc_bias_key in checkpoint: del checkpoint[fc_bias_key]
                model_to_finetune.load_state_dict(checkpoint, strict=False)
                self.logger.info(f"Loaded backbone weights. Final layer '{final_layer_name}' will be trained from scratch.")

            else: # If final layer matches or doesn't exist in checkpoint with that name
                model_to_finetune.load_state_dict(checkpoint, strict=False)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        optimizer = torch.optim.AdamW(model_to_finetune.parameters(), lr=self.config.learning_rate)

        # train_model now returns a tuple (trained_model_instance, metrics_dict)
        # We need the model instance that was actually trained and had its state loaded.
        trained_model_instance, _ = train_model(
            model=model_to_finetune, # Pass the model instance that's on the correct device
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer, # This will be used as a template by train_model
            n_splits=self._get_n_splits_for_finetune(),
            n_runs=30, # Consider making n_runs configurable via TrainingConfig
            num_epochs=self.config.epochs,
            patience=self.config.early_stopping,
            is_augmented=self.config.data_augmentation, # train_model will handle AugmentationConfig creation
            device=self.device
        )
        self.logger.info("Main fine-tuning finished.")
        # Optionally save the fine-tuned model here
        # torch.save(trained_model_instance.state_dict(), Path(self.config.output).parent / f"{self.config.model_type}_{self.config.dataset}_run{self.config.run}_final.pth")
        return trained_model_instance

    def _create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create model instance based on self.model_type."""
        # This long if/elif is kept for clarity due to varying model constructor args.
        # A dict-based factory is an alternative for fewer lines but more setup.
        model_args_common = {"dropout": self.config.dropout}
        if self.model_type == "transformer":
            model = Transformer(input_dim=input_dim, output_dim=output_dim,
                                num_layers=self.config.num_layers, num_heads=self.config.num_heads,
                                hidden_dim=self.config.hidden_dimension, **model_args_common)
        elif self.model_type == "lstm":
            model = LSTM(input_size=input_dim, output_size=output_dim,
                         hidden_size=self.config.hidden_dimension, num_layers=self.config.num_layers,
                         **model_args_common)
        elif self.model_type == "cnn":
            model = CNN(input_size=input_dim, num_classes=output_dim, **model_args_common)
        elif self.model_type == "rcnn": # Assuming RCNN takes input_size, num_classes
            model = RCNN(input_size=input_dim, num_classes=output_dim, **model_args_common)
        elif self.model_type == "mamba":
            model = Mamba(d_model=input_dim, n_classes=output_dim, d_state=self.config.hidden_dimension,
                          d_conv=4, expand=2, depth=self.config.num_layers) # Mamba might not use dropout arg directly
        elif self.model_type == "kan":
            model = KAN(input_dim=input_dim, output_dim=output_dim, hidden_dim=self.config.hidden_dimension,
                        num_layers=self.config.num_layers, dropout_rate=self.config.dropout, num_inner_functions=10)
        elif self.model_type == "vae":
            model = VAE(input_size=input_dim, num_classes=output_dim, latent_dim=self.config.hidden_dimension, **model_args_common)
        elif self.model_type == "moe":
            model = MOE(input_dim=input_dim, output_dim=output_dim, num_heads=self.config.num_heads,
                        num_layers=self.config.num_layers, hidden_dim=self.config.hidden_dimension,
                        num_experts=4, k=2) # MOE specific args
        elif self.model_type == "dense":
            model = Dense(input_dim=input_dim, output_dim=output_dim, **model_args_common)
        elif self.model_type == "ode": # Assuming ODE takes these
            model = ODE(input_dim=input_dim, output_dim=output_dim, **model_args_common)
        elif self.model_type == "rwkv": # Assuming RWKV takes these
             model = RWKV(input_dim=input_dim, output_dim=output_dim, hidden_dim=self.config.hidden_dimension, **model_args_common)
        elif self.model_type == "tcn": # Assuming TCN takes these
            model = TCN(input_dim=input_dim, output_dim=output_dim, **model_args_common)
        elif self.model_type == "wavenet": # Assuming WaveNet takes these
            model = WaveNet(input_dim=input_dim, output_dim=output_dim, **model_args_common)
        elif self.model_type == "ensemble": # Assuming Ensemble takes these
            model = Ensemble(input_dim=input_dim, output_dim=output_dim, hidden_dim=self.config.hidden_dimension, **model_args_common)
        elif self.model_type == "diffusion":
            model = Diffusion(input_dim=input_dim, output_dim=output_dim, hidden_dim=self.config.hidden_dimension,
                              time_dim=64, num_timesteps=1000)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        self.logger.info(f"Created model: {self.model_type} instance.")
        return model

    def _calculate_class_weights(self, train_loader: DataLoader) -> Optional[torch.Tensor]:
        # This method is not currently called in the flow, but kept for potential use.
        # Ensure labels are class indices for this calculation.
        class_counts = torch.zeros(self.n_classes, dtype=torch.float)
        for _, labels_batch in train_loader: # Assuming labels are class indices
            if labels_batch.dim() > 1 and labels_batch.shape[-1] > 1: # One-hot
                labels_batch = labels_batch.argmax(dim=-1)
            for label_idx in labels_batch:
                if 0 <= label_idx.item() < self.n_classes:
                    class_counts[label_idx.item()] += 1
        
        if torch.any(class_counts == 0):
            self.logger.warning("Some classes have zero samples in training data; class weights might be problematic.")
            # Handle missing classes, e.g., assign a very small count or skip weighting.
            return None # Or return uniform weights
            
        weights = 1.0 / class_counts
        return (weights / weights.sum() * self.n_classes).to(self.device)

    def _plot_attention_maps(self, model: nn.Module, data: Any) -> None: # model type might not be Transformer always
        self.logger.warning("_plot_attention_maps is not implemented.")
        # raise NotImplementedError # Original was raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog="ModelTraining", description="Spectra Model Training Pipeline.")
    # File/Dataset
    parser.add_argument("-f", "--file-path", type=str, default="model_checkpoint", help="Base path for model checkpoints")
    parser.add_argument("-d", "--dataset", type=str, default="species", choices=ModelTrainer.N_CLASSES_PER_DATASET.keys(), help="Dataset name")
    parser.add_argument("-m", "--model", type=str, default="transformer", help="Model type") # Add choices later if MODEL_REGISTRY is used
    parser.add_argument("-r", "--run", type=int, default=0, help="Run identifier (for logging/output naming)")
    parser.add_argument("-o", "--output", type=str, default="logs/results_base", help="Base name for output logs/results")
    # Augmentation general
    parser.add_argument("-da", "--data-augmentation", action="store_true", help="Enable data augmentation")
    # Pre-training Task Flags (iterating PRETRAIN_TASK_DEFINITIONS to create these)
    for task_flag, _, task_desc_suffix, _, _ in ModelTrainer.PRETRAIN_TASK_DEFINITIONS:
        parser.add_argument(f"--{task_flag.replace('_', '-')}", action="store_true", help=f"Enable {task_desc_suffix.replace('_', ' ')}")
    # Regularization
    parser.add_argument("-es", "--early-stopping", type=int, default=20, help="Early stopping patience")
    parser.add_argument("-do", "--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("-ls", "--label-smoothing", type=float, default=0.1, help="Label smoothing alpha")
    # Hyperparameters
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="Learning rate") # Adjusted default
    parser.add_argument("-bs", "--batch-size", type=int, default=64, help="Batch size") # Adjusted default
    parser.add_argument("-hd", "--hidden-dimension", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("-l", "--num-layers", type=int, default=4, help="Number of layers (model-specific meaning)")
    parser.add_argument("-nh", "--num-heads", type=int, default=4, help="Number of attention heads (for Transformer)")
    # Augmentation specific params
    parser.add_argument("--num-augmentations", type=int, default=5, help="Augmentations per sample")
    parser.add_argument("--noise-level", type=float, default=0.1, help="Noise level for augmentation")
    parser.add_argument("--shift-enabled", action="store_true", help="Enable shift augmentation")
    parser.add_argument("--scale-enabled", action="store_true", help="Enable scale augmentation")
    return parser.parse_args()

def main() -> None:
    """Main execution function."""
    trainer_instance = None # For access in except block
    try:
        args = parse_arguments()
        config = TrainingConfig.from_args(args)
        
        trainer_instance = ModelTrainer(config)
        logger = trainer_instance.logger # Use instance logger
        logger.info(f"Training configuration: {config}")
        logger.info(f"Using device: {trainer_instance.device}")
        logger.info("Starting training pipeline")

        any_pretrain_task_enabled = any(
            getattr(config, task_def[0], False) for task_def in ModelTrainer.PRETRAIN_TASK_DEFINITIONS
        )
        pre_trained_model_output = None
        if any_pretrain_task_enabled:
            logger.info("Pre-training tasks enabled. Setting up pre-training data module.")
            # Assuming create_data_module can handle TrainingConfig or relevant fields for augmentation
            pretrain_data_module = create_data_module(
                dataset_name=config.dataset, batch_size=config.batch_size,
                augmentation_config=config, # Pass the whole config or specific aug fields
                is_pre_train=True
            )
            trainer_instance.data_module = pretrain_data_module
            pre_trained_model_output = trainer_instance.pre_train()
        else:
            logger.info("No pre-training tasks enabled. Skipping pre-training phase.")

        logger.info("Setting up main fine-tuning data module.")
        main_data_module = create_data_module(
            # file_path="/home/woodj/Desktop/fishy-business/data/REIMS.xlsx",  # Assuming this is needed for main training
            # file_path = "/vol/ecrg-solar/woodj4/fishy-business/data/REIMS.xlsx" # Example server path
            file_path="/Users/woodj/Desktop/fishy-business/data/REIMS.xlsx",  # Example local path
            dataset_name=config.dataset, batch_size=config.batch_size,
            augmentation_config=config, # Pass the whole config or specific aug fields
            is_pre_train=False
        )
        trainer_instance.data_module = main_data_module
        
        final_trained_model = trainer_instance.train(pre_trained_model_output)
        logger.info(f"Training pipeline completed. Final model type: {type(final_trained_model)}")

    except Exception as e:
        current_logger = getattr(trainer_instance, 'logger', logging.getLogger(__name__))
        current_logger.error(f"Critical error in training pipeline: {str(e)}", exc_info=True)
        # traceback.print_exc() # For console output during dev if logger isn't capturing well
        # Consider exiting with an error code: sys.exit(1)
        raise # Re-raise the exception after logging

if __name__ == "__main__":
    main()
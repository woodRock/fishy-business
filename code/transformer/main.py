import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from plot import plot_attention_map
from pre_training import PreTrainer, PreTrainingConfig
from transformer import Transformer
from train import train_model
from util import preprocess_dataset, create_data_module, AugmentationConfig

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # File and dataset settings
    file_path: str = "transformer_checkpoint.pth"
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
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        """Create configuration from command line arguments."""
        return cls(
            file_path=args.file_path,
            dataset=args.dataset,
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
            scale_enabled=args.scale_enabled
        )

class ModelTrainer:
    """Handles the complete training pipeline including pre-training and fine-tuning."""
    
    N_CLASSES_PER_DATASET = {
        "species": 2,
        "part": 7,
        "oil": 7,
        "cross-species": 3,
        "instance-recognition": 2,
        "instance-recognition-hard": 24
    }
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.logger = self._setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if config.dataset not in self.N_CLASSES_PER_DATASET:
            raise ValueError(f"Invalid dataset: {config.dataset}")
            
        self.n_classes = self.N_CLASSES_PER_DATASET[config.dataset]
        self.n_features = 2080  # Could be made configurable if needed
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        output = f"{self.config.output}_{self.config.run}.log"
        logging.basicConfig(
            filename=output,
            level=logging.INFO,
            filemode='w',
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logger

    def pre_train(self) -> Optional[Transformer]:
        """Run pre-training if configured."""
        if not (self.config.masked_spectra_modelling or self.config.next_spectra_prediction):
            return None
            
        self.logger.info("Starting pre-training phase")
        
        # Load pre-training data
        train_loader, data = preprocess_dataset(
            self.config.dataset,
            self.config.data_augmentation,
            batch_size=self.config.batch_size,
            is_pre_train=True
        )
        
        # Initialize model for pre-training
        model = self._create_model(input_dim=2080, output_dim=2080)
        model = model.to(self.device)
        
        # Setup pre-training configuration
        pre_training_config = PreTrainingConfig(
            num_epochs=self.config.epochs,
            file_path=self.config.file_path,
            device=self.device
        )
        
        # Initialize pre-trainer
        pre_trainer = PreTrainer(
            model=model,
            config=pre_training_config,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        )
        
        # Masked Spectra Modelling
        if self.config.masked_spectra_modelling:
            self.logger.info("Starting Masked Spectra Modelling")
            start_time = time.time()
            model = pre_trainer.pre_train_masked_spectra(train_loader)
            self.logger.info(f"MSM training time: {time.time() - start_time:.2f}s")
        
        # Next Spectra Prediction
        if self.config.next_spectra_prediction:
            self.logger.info("Starting Next Spectra Prediction")
            start_time = time.time()
            # Reinitialize model for NSP if needed
            if self.config.masked_spectra_modelling:
                model = self._create_model(input_dim=1023, output_dim=2)
                pre_trainer = PreTrainer(
                    model=model,
                    config=pre_training_config,
                    criterion=nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing),
                    optimizer=torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
                )
            
            val_loader, _ = preprocess_dataset(
                self.config.dataset,
                self.config.data_augmentation,
                batch_size=self.config.batch_size,
                is_pre_train=True
            )
            model = pre_trainer.pre_train_next_spectra(train_loader, val_loader)
            self.logger.info(f"NSP training time: {time.time() - start_time:.2f}s")
        
        return model

    def train(self, pre_trained_model: Optional[Transformer] = None) -> Transformer:
        """Run main training phase."""
        self.logger.info("Starting main training phase")
        
        # Load training data
        train_loader, data = preprocess_dataset(
            self.config.dataset,
            self.config.data_augmentation,
            batch_size=self.config.batch_size,
            is_pre_train=False
        )
        
        # Calculate class weights
        class_weights = self._calculate_class_weights(train_loader)
        class_weights = class_weights.to(self.device)
        
        # Initialize model
        model = self._create_model(self.n_features, self.n_classes)
        
        # Transfer weights if pre-trained
        if pre_trained_model is not None:
            self.logger.info("Transferring pre-trained weights")
            model.load_state_dict(pre_trained_model.state_dict(), strict=False)
        
        model = model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.config.label_smoothing
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        
        # Train model
        start_time = time.time()
        model = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            num_epochs=self.config.epochs,
            patience=self.config.early_stopping
        )
        self.logger.info(f"Training time: {time.time() - start_time:.2f}s")
        
        # Plot attention maps
        self._plot_attention_maps(model, data)
        
        return model
    
    def _create_model(self, input_dim: int, output_dim: int) -> Transformer:
        """Create a new transformer model instance."""
        model = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dimension,
            dropout=self.config.dropout
        )
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
        i = 10
        columns = data.axes[1][1:(i+1)].tolist()
        
        # First self-attention layer of the encoder
        attention_weights = model.encoder.layers[0].self_attention.fc_out.weight
        attention_weights = attention_weights[:i,:i].cpu().detach().numpy()
        plot_attention_map("encoder", attention_weights, columns, columns)
        
        # Last self-attention layer of the decoder
        attention_weights = model.decoder.layers[-1].self_attention.fc_out.weight
        attention_weights = attention_weights[:i,:i].cpu().detach().numpy()
        plot_attention_map("decoder", attention_weights, columns, columns)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='Transformer',
        description='A transformer for fish species classification.'
    )
    
    # File and dataset settings
    parser.add_argument('-f', '--file-path', type=str, default="transformer_checkpoint",
                    help="Filepath for model checkpoints")
    parser.add_argument('-d', '--dataset', type=str, default="species",
                    help="Fish species or part dataset")
    parser.add_argument('-r', '--run', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default="logs/results")
    
    # Preprocessing flags
    parser.add_argument('-da', '--data-augmentation', action='store_true',
                    help="Enable data augmentation")
    parser.add_argument('-msm', '--masked-spectra-modelling', action='store_true',
                    help="Enable masked spectra modelling")
    parser.add_argument('-nsp', '--next-spectra-prediction', action='store_true',
                    help="Enable next spectra prediction")
    
    # Regularization settings
    parser.add_argument('-es', '--early-stopping', type=int, default=10,
                    help='Early stopping patience')
    parser.add_argument('-do', '--dropout', type=float, default=0.2,
                    help="Dropout probability")
    parser.add_argument('-ls', '--label-smoothing', type=float, default=0.1,
                    help="Label smoothing alpha value")
    
    # Training hyperparameters
    parser.add_argument('-e', '--epochs', type=int, default=100,
                    help="Number of training epochs")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5,
                    help="Learning rate")
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                    help='Batch size')
    parser.add_argument('-hd', '--hidden-dimension', type=int, default=128,
                    help="Hidden dimension size")
    parser.add_argument('-l', '--num-layers', type=float, default=4,
                    help="Number of transformer layers")
    parser.add_argument('-nh', '--num-heads', type=int, default=4,
                    help='Number of attention heads')
    
    # Data augmentation parameters
    parser.add_argument('--num-augmentations', type=int, default=5,
                    help="Number of augmentations per sample")
    parser.add_argument('--noise-level', type=float, default=0.1,
                    help="Level of noise for augmentation")
    parser.add_argument('--shift-enabled', action='store_true',
                    help="Enable shift augmentation")
    parser.add_argument('--scale-enabled', action='store_true',
                    help="Enable scale augmentation")
    
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
        
        # Create augmentation configuration
        aug_config = AugmentationConfig(
            enabled=config.data_augmentation,
            num_augmentations=config.num_augmentations,
            noise_enabled=True,  # Always enable noise for augmentation
            shift_enabled=config.shift_enabled,
            scale_enabled=config.scale_enabled,
            noise_level=config.noise_level
        )
        
        # Create main training data module
        data_module = create_data_module(
            dataset_name=config.dataset,
            batch_size=config.batch_size,
            augmentation_enabled=config.data_augmentation,
            is_pre_train=False,
            **{
                'num_augmentations': config.num_augmentations,
                'noise_level': config.noise_level,
                'shift_enabled': config.shift_enabled,
                'scale_enabled': config.scale_enabled
            }
        )
        
        # Setup main data module to get loaders
        train_loader, data = data_module.setup()
        trainer.logger.info(f"Main training dataset size: {len(train_loader.dataset)} samples")
        
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
                    'num_augmentations': config.num_augmentations,
                    'noise_level': config.noise_level,
                    'shift_enabled': config.shift_enabled,
                    'scale_enabled': config.scale_enabled
                }
            )
            
            # Setup pre-training data module
            pretrain_loader, _ = pretrain_data_module.setup()
            trainer.logger.info(f"Pre-training dataset size: {len(pretrain_loader.dataset)} samples")
            
            # Update trainer's data module for pre-training
            trainer.data_module = pretrain_data_module
            pre_trained_model = trainer.pre_train()
            
            if pre_trained_model is not None:
                trainer.logger.info("Pre-training completed successfully")
        
        # Update trainer's data module for main training
        trainer.data_module = data_module
        
        # Run main training
        model = trainer.train(pre_trained_model)
        
        # Log final dataset information
        trainer.logger.info("Training pipeline completed successfully")
        trainer.logger.info(f"Final training dataset size: {len(train_loader.dataset)} samples")
        trainer.logger.info(f"Original dataset shape: {data.shape}")
        
        if config.data_augmentation:
            trainer.logger.info(
                f"Augmentation ratio: {len(train_loader.dataset) / data.shape[0]:.2f}x "
                f"(Expected: {config.num_augmentations + 1}x)"
            )
        
        return model
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
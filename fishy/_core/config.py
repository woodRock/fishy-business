# -*- coding: utf-8 -*-
"""
Configuration module for the deep learning training pipeline.

This module defines the :class:`TrainingConfig` dataclass, which centralizes all configuration parameters
for model training, data loading, and augmentation. It also provides a method to create a configuration
instance directly from parsed command-line arguments.
"""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Configuration for model training, combining settings from command-line arguments.

    Examples:
        >>> # Default configuration
        >>> cfg = TrainingConfig(model="transformer", dataset="species")
        >>> cfg.model
        'transformer'
        >>> cfg.epochs
        100
    """

    file_path: str = ""
    model: str = "transformer"
    dataset: str = "species"
    run: int = 0
    output: str = "logs/results"
    data_augmentation: bool = False
    masked_spectra_modelling: bool = False
    next_spectra_prediction: bool = False
    next_peak_prediction: bool = False
    spectrum_denoising_autoencoding: bool = False
    peak_parameter_regression: bool = False
    spectrum_segment_reordering: bool = False
    contrastive_transformation_invariance_learning: bool = False
    early_stopping: int = 20
    dropout: float = 0.2
    label_smoothing: float = 0.1
    epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 64
    hidden_dimension: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_augmentations: int = 5
    noise_level: float = 0.05
    shift_enabled: bool = False
    scale_enabled: bool = False
    k_folds: int = 3
    num_runs: int = 1
    use_coral: bool = False
    use_cumulative_link: bool = False
    regression: bool = False
    use_groups: bool = False # Added flag for Group-Aware Splitting
    # New augmentation parameters
    crop_enabled: bool = False
    flip_enabled: bool = False
    permutation_enabled: bool = False
    crop_size: float = 0.8  # Default crop size

    # Weights & Biases parameters
    wandb_project: Optional[str] = "fishy-business"
    wandb_entity: Optional[str] = "victoria-university-of-wellington"
    wandb_log: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """
        Create a :class:`TrainingConfig` instance from parsed command-line arguments,
        merging with model-specific defaults from models.yaml.
        """
        import dataclasses
        from fishy._core.config_loader import load_config

        # 1. Start with class defaults
        config_dict = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
        
        # 2. Load model-specific defaults from YAML if available
        if hasattr(args, "model") and args.model:
            models_cfg = load_config("models")["deep_models"]
            model_entry = models_cfg.get(args.model.lower())
            if isinstance(model_entry, dict) and "defaults" in model_entry:
                config_dict.update(model_entry["defaults"])

        # 3. Override with explicitly provided command-line arguments
        # We only override if the argument was actually passed by the user
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        arg_dict = vars(args)
        for key in valid_keys:
            if key in arg_dict and arg_dict[key] is not None:
                # For boolean flags, check if they are True (explicitly enabled)
                # For other types, check if they differ from the default if default is set
                config_dict[key] = arg_dict[key]

        return cls(**config_dict)

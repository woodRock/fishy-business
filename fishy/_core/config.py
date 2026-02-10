# -*- coding: utf-8 -*-
"""
Configuration module for the deep learning training pipeline.
"""

import argparse
from dataclasses import dataclass

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
    k_folds: int
    num_runs: int = 1
    use_coral: bool = False
    use_cumulative_link: bool = False
    regression: bool = False
    # New augmentation parameters
    crop_enabled: bool = False
    flip_enabled: bool = False
    permutation_enabled: bool = False
    crop_size: float = 0.8  # Default crop size

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create configuration from command line arguments."""
        return cls(**vars(args))

# -*- coding: utf-8 -*-
"""
Configuration module for the deep learning training pipeline.

This module defines the :class:`TrainingConfig` dataclass, which centralizes all configuration parameters
for model training, data loading, and augmentation. It also provides a method to create a configuration
instance directly from parsed command-line arguments.
"""

import argparse
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Configuration for model training, combining settings from command-line arguments.

    This dataclass holds all the hyperparameters and settings required to configure the
    training pipeline, including dataset paths, model choices, training parameters (epochs, learning rate),
    regularization techniques (dropout, label smoothing), and data augmentation options.

    Attributes:
        file_path (str): Path to the dataset file (e.g., an Excel or CSV file).
        model (str): Name of the model architecture to use (e.g., 'transformer', 'cnn').
        dataset (str): Name of the dataset to load (e.g., 'species', 'part').
        run (int): Identifier for the current experimental run (useful for logging and seeding).
        output (str): Base path for outputting logs and results.
        data_augmentation (bool): Whether to enable data augmentation.
        masked_spectra_modelling (bool): Enable Masked Spectra Modelling pre-training task.
        next_spectra_prediction (bool): Enable Next Spectra Prediction pre-training task.
        next_peak_prediction (bool): Enable Next Peak Prediction pre-training task.
        spectrum_denoising_autoencoding (bool): Enable Spectrum Denoising Autoencoding pre-training task.
        peak_parameter_regression (bool): Enable Peak Parameter Regression pre-training task.
        spectrum_segment_reordering (bool): Enable Spectrum Segment Reordering pre-training task.
        contrastive_transformation_invariance_learning (bool): Enable Contrastive Transformation Invariance Learning pre-training task.
        early_stopping (int): Number of epochs to wait for improvement before stopping training early.
        dropout (float): Dropout probability used in the model.
        label_smoothing (float): Label smoothing factor for cross-entropy loss.
        epochs (int): Maximum number of training epochs.
        learning_rate (float): Initial learning rate for the optimizer.
        batch_size (int): Batch size for training and validation.
        hidden_dimension (int): Size of the hidden layers in the model.
        num_layers (int): Number of layers in the model (e.g., Transformer blocks).
        num_heads (int): Number of attention heads (for Transformer-based models).
        num_augmentations (int): Number of augmented samples to generate per original sample.
        noise_level (float): Magnitude of noise to add during augmentation.
        shift_enabled (bool): Whether to enable shift augmentation.
        scale_enabled (bool): Whether to enable scale augmentation.
        k_folds (int): Number of folds for k-fold cross-validation.
        num_runs (int): Number of independent training runs to perform (default: 1).
        use_coral (bool): Whether to use CORAL loss for ordinal regression.
        use_cumulative_link (bool): Whether to use Cumulative Link loss for ordinal regression.
        regression (bool): Whether to treat the task as a standard regression problem.
        crop_enabled (bool): Whether to enable random cropping augmentation (default: False).
        flip_enabled (bool): Whether to enable random flipping augmentation (default: False).
        permutation_enabled (bool): Whether to enable random permutation augmentation (default: False).
        crop_size (float): The proportion of the spectrum to keep during random cropping (default: 0.8).
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
    # New augmentation parameters
    crop_enabled: bool = False
    flip_enabled: bool = False
    permutation_enabled: bool = False
    crop_size: float = 0.8  # Default crop size

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """
        Create a :class:`TrainingConfig` instance from parsed command-line arguments.

        Args:
            args (argparse.Namespace): The parsed arguments from :mod:`argparse`.

        Returns:
            TrainingConfig: A configuration object populated with values from ``args``.
        """
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        config_dict = {k: v for k, v in vars(args).items() if k in valid_keys}
        return cls(**config_dict)
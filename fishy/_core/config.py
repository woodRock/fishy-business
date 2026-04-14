# -*- coding: utf-8 -*-
"""
Configuration module for the deep learning training pipeline.
"""

import argparse
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration for model training, combining settings from command-line arguments.

    Examples:
        >>> config = TrainingConfig(model="cnn", batch_size=32)
        >>> config.model
        'cnn'
        >>> config.batch_size
        32
    """

    file_path: Optional[str] = None
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
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    num_augmentations: int = 5
    noise_level: float = 0.05
    shift_enabled: bool = False
    scale_enabled: bool = False
    k_folds: int = 3
    num_runs: int = 1
    ordinal_method: Optional[str] = None  # coral, clm
    regression: bool = False
    use_groups: bool = False  # Added flag for Group-Aware Splitting
    # New augmentation parameters
    crop_enabled: bool = False
    flip_enabled: bool = False
    permutation_enabled: bool = False
    crop_size: float = 0.8  # Default crop size
    num_experts: int = 8
    k: int = 2
    random_projection: bool = False,
    quantize: bool = False,
    turbo_quant: bool = False,
    polar: bool = False,
    normalize: bool = False,
    snv: bool = False,
    minmax: bool = False,
    log_transform: bool = False,
    savgol: bool = False,


    # Deep Model Specific
    top_k: Optional[int] = None
    use_performer: bool = False
    use_checkpointing: bool = False
    use_xsa: bool = False
    # ... other model params

    # Contrastive specific
    encoder_type: str = "dense"

    # New Analysis & Reporting Flags
    benchmark: bool = False
    figures: bool = False
    xai: bool = False
    statistical: bool = False

    # Task specific (now integrated)
    method: str = "deep"  # deep, classic, contrastive, evolutionary
    transfer: bool = False
    transfer_datasets: Optional[list] = None
    target_dataset: Optional[str] = None
    epochs_transfer: int = 10
    epochs_finetune: int = 20

    # Weights & Biases parameters
    wandb_project: Optional[str] = "fishy-business"
    wandb_entity: Optional[str] = "victoria-university-of-wellington"
    wandb_log: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create from CLI arguments."""
        import dataclasses
        from fishy._core.config_loader import load_config

        # 1. Start with class defaults
        config_dict = {
            f.name: f.default
            for f in dataclasses.fields(cls)
            if f.default is not dataclasses.MISSING
        }

        # 2. Load model-specific defaults from YAML if available
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        if hasattr(args, "model") and args.model:
            models_cfg = load_config("models")["deep_models"]
            model_entry = models_cfg.get(args.model.lower())
            if isinstance(model_entry, dict) and "defaults" in model_entry:
                model_defaults = model_entry["defaults"].copy()
                # Normalise hidden_dimension -> hidden_dim
                if "hidden_dimension" in model_defaults:
                    model_defaults["hidden_dim"] = model_defaults.pop(
                        "hidden_dimension"
                    )
                # Only apply keys that TrainingConfig actually knows about
                config_dict.update(
                    {k: v for k, v in model_defaults.items() if k in valid_keys}
                )

        # 3. Override with explicitly provided command-line arguments
        arg_dict = vars(args)

        # Handle special mappings
        if "ordinal" in arg_dict and arg_dict["ordinal"]:
            config_dict["ordinal_method"] = arg_dict["ordinal"]

        if "encoder" in arg_dict and arg_dict["encoder"]:
            config_dict["encoder_type"] = arg_dict["encoder"]

        for key in valid_keys:
            if key in arg_dict and arg_dict[key] is not None:
                config_dict[key] = arg_dict[key]

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Loads configuration from a YAML file."""
        import dataclasses

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Filter out keys that are not fields of the dataclass
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_yaml(self, path: str):
        """Saves configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


@dataclass
class ExperimentConfig:
    """
    Configuration for a batch of experiments.

    Examples:
        >>> exp = ExperimentConfig(name="test_suite", models=["cnn", "transformer"])
        >>> exp.name
        'test_suite'
        >>> exp.models
        ['cnn', 'transformer']
    """

    name: str = "batch_experiment"
    num_runs: int = 1
    datasets: List[str] = field(default_factory=lambda: ["species"])
    models: List[str] = field(default_factory=lambda: ["transformer"])
    # Common overrides for all runs in this experiment
    overrides: Dict[str, Any] = field(default_factory=dict)

    # Analysis flags for the batch
    benchmark: bool = False
    figures: bool = False
    wandb_log: bool = False
    statistical: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

# -*- coding: utf-8 -*-
"""
Model factory for creating deep learning models.

This module provides a centralized mechanism for instantiating various deep learning models
based on a configuration object. It includes a registry of available models and helper functions
to handle specific initialization requirements, such as wrapping models for Siamese networks
used in instance recognition tasks.
"""

import importlib
import torch
import torch.nn as nn
from typing import Dict, Type, Any

from fishy._core.config import TrainingConfig
from fishy._core.config_loader import load_config


def get_model_class(model_path: str) -> Type[nn.Module]:
    """
    Dynamically imports a model class from a string path.

    Args:
        model_path (str): The full dot-separated path to the model class
                          (e.g., "fishy.models.deep.transformer.Transformer").

    Returns:
        Type[nn.Module]: The imported model class.

    Examples:
        >>> cls = get_model_class("fishy.models.deep.transformer.Transformer")
        >>> cls.__name__
        'Transformer'
    """
    module_path, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_model(config: TrainingConfig, input_dim: int, output_dim: int) -> nn.Module:
    """
    Factory function to create a deep learning model based on dynamic configuration.

    It reads the model registry from `models.yaml`, resolves the class path,
    and instantiates the model with parameters from `TrainingConfig`.

    Args:
        config (TrainingConfig): The experiment configuration object.
        input_dim (int): Dimensionality of input features.
        output_dim (int): Number of output classes (or dimensions).

    Returns:
        nn.Module: The instantiated PyTorch model.

    Raises:
        ValueError: If the model name is not found in the registry.
        RuntimeError: If model instantiation fails due to signature mismatch.

    Examples:
        >>> from fishy._core.config import TrainingConfig
        >>> cfg = TrainingConfig(model="transformer", dataset="species", hidden_dimension=64, num_layers=2)
        >>> model = create_model(cfg, input_dim=1024, output_dim=10)
        >>> isinstance(model, nn.Module)
        True
    """
    model_name = config.model.lower()
    models_cfg = load_config("models")["deep_models"]

    if model_name not in models_cfg:
        raise ValueError(
            f"Model '{model_name}' not found in registry. Available: {list(models_cfg.keys())}"
        )

    entry = models_cfg[model_name]
    model_path = entry["path"] if isinstance(entry, dict) else entry
    model_class = get_model_class(model_path)

    # Handle Siamese wrappers for instance-recognition
    if "instance-recognition" in config.dataset:
        # Generic wrapper logic could be added here
        # For now, keep the specialized ones but using dynamic classes
        if model_name == "mamba":
            from fishy.models.deep.mamba import SiameseMamba

            return SiameseMamba(
                input_dim, output_dim, config.hidden_dimension, config.num_layers
            )
        elif model_name == "vae":
            from fishy.models.deep.vae import SiameseVAE

            vae_backbone = model_class(
                input_size=input_dim,
                latent_dim=config.hidden_dimension,
                num_classes=output_dim,
                dropout=config.dropout,
            )
            return SiameseVAE(vae_backbone)

    # Dynamic instantiation based on common signature patterns
    # We try different common signatures used in this project
    try:
        if model_name == "transformer":
            return model_class(
                input_dim,
                output_dim,
                config.num_heads,
                config.hidden_dimension,
                config.num_layers,
                config.dropout,
            )
        elif model_name == "lstm":
            return model_class(
                input_dim,
                config.hidden_dimension,
                config.num_layers,
                output_dim,
                config.dropout,
            )
        elif model_name == "mamba":
            return model_class(
                input_dim,
                output_dim,
                config.hidden_dimension,
                16,
                4,
                2,
                config.num_layers,
                config.dropout,
            )
        elif model_name == "vae":
            return model_class(
                input_size=input_dim,
                latent_dim=config.hidden_dimension,
                num_classes=output_dim,
                dropout=config.dropout,
            )
        elif model_name == "ensemble":
            return model_class(
                input_dim, config.hidden_dimension, output_dim, config.dropout
            )
        elif model_name == "moe":
            return model_class(
                input_dim, output_dim, config.hidden_dimension, config.num_layers
            )
        elif model_name == "rwkv":
            return model_class(
                input_dim, config.hidden_dimension, output_dim, config.dropout
            )

        # Fallback for simpler models
        return model_class(input_dim, output_dim)
    except Exception as e:
        # If specific signature failed, try a very generic one or re-raise
        try:
            return model_class(input_dim, output_dim)
        except:
            raise RuntimeError(f"Failed to instantiate model '{model_name}': {e}")

# -*- coding: utf-8 -*-
"""
Model factory for creating deep learning models with unified signatures.
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
    """
    module_path, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_model(config: TrainingConfig, input_dim: int, output_dim: int) -> nn.Module:
    """
    Unified factory function to create a deep learning model.
    All models now follow a standard (input_dim, output_dim, hidden_dim, num_layers, dropout) signature.
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

    # Standardized parameters from config
    params = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": config.hidden_dimension,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "num_heads": config.num_heads
    }

    # Handle Siamese wrappers for instance-recognition
    if "instance-recognition" in config.dataset:
        if model_name == "vae":
            from fishy.models.deep.vae import SiameseVAE
            backbone = model_class(**params)
            return SiameseVAE(backbone)
        
        # Generic Siamese wrapper for other models could be instantiated here
        # For now, we use specialized ones if they exist, or a default fallback
        try:
            # Check if there's a Siamese version in the same module
            siamese_class = get_model_class(model_path.replace(model_class.__name__, f"Siamese{model_class.__name__}"))
            return siamese_class(**params)
        except:
            # Fallback to standard model if no Siamese wrapper found
            return model_class(**params)

    # Generic instantiation for all unified models
    return model_class(**params)

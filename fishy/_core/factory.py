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
from fishy._core.constants import DatasetName


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
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "top_k": config.top_k,
        "use_performer": config.use_performer,
        "use_checkpointing": config.use_checkpointing,
        "use_xsa": config.use_xsa,
        "use_qk_gain": config.use_qk_gain,
        "use_parallel_residuals": config.use_parallel_residuals,
        "recurrence_layers": config.recurrence_layers,
        "use_leaky_sq": config.use_leaky_sq,
        "use_post_norm": config.use_post_norm,
        "logit_cap": config.logit_cap,
    }

    # Handle Siamese wrappers for batch-detection
    if DatasetName.BATCH_DETECTION in config.dataset:
        if model_name == "vae":
            from fishy.models.deep.vae import SiameseVAE

            backbone = model_class(**params)
            return SiameseVAE(backbone)

        # Generic Siamese wrapper for other models could be instantiated here
        # For now, we use specialized ones if they exist, or a default fallback
        try:
            # Check if there's a Siamese version in the same module
            siamese_class = get_model_class(
                model_path.replace(
                    model_class.__name__, f"Siamese{model_class.__name__}"
                )
            )
            return siamese_class(**params)
        except (ImportError, AttributeError):
            # Fallback to standard model if no Siamese wrapper found
            return model_class(**params)

    # Generic instantiation for all unified models
    return model_class(**params)

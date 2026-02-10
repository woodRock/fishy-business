# -*- coding: utf-8 -*-
"""
Model factory for creating deep learning models.

This module provides a centralized mechanism for instantiating various deep learning models
based on a configuration object. It includes a registry of available models and helper functions
to handle specific initialization requirements, such as wrapping models for Siamese networks
used in instance recognition tasks.
"""

import torch
import torch.nn as nn
from typing import Dict, Type

# Absolute imports from specific modules to avoid circularity through __init__
from fishy.models.deep.transformer import Transformer
from fishy.models.deep.lstm import LSTM
from fishy.models.deep.cnn import CNN
from fishy.models.deep.rcnn import RCNN
from fishy.models.deep.mamba import Mamba, SiameseMamba
from fishy.models.deep.kan import KAN
from fishy.models.deep.vae import VAE, SiameseVAE
from fishy.models.deep.MOE import MOE
from fishy.models.deep.dense import Dense
from fishy.models.deep.ode import ODE
from fishy.models.deep.rwkv import RWKV
from fishy.models.deep.tcn import TCN
from fishy.models.deep.wavenet import WaveNet
from fishy.models.deep.hybrid import Hybrid
from fishy.models.deep.performer import Performer
from fishy.models.deep.ordinal import TransformerOrdinal
from fishy.models.deep.ensemble import Ensemble

from fishy._core.config import TrainingConfig

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "transformer": Transformer,
    "lstm": LSTM,
    "cnn": CNN,
    "rcnn": RCNN,
    "mamba": Mamba,
    "kan": KAN,
    "vae": VAE,
    "moe": MOE,
    "dense": Dense,
    "ode": ODE,
    "rwkv": RWKV,
    "tcn": TCN,
    "wavenet": WaveNet,
    "hybrid": Hybrid,
    "performer": Performer,
    "ordinal": TransformerOrdinal,
    "ensemble": Ensemble,
}


def create_model(config: TrainingConfig, input_dim: int, output_dim: int) -> nn.Module:
    """
    Factory function to create a model based on the configuration.

    Args:
        config (TrainingConfig): Configuration object containing model parameters.
        input_dim (int): Input feature dimension.
        output_dim (int): Output feature/class dimension.

    Returns:
        nn.Module: The instantiated and configured PyTorch model.
    """
    model_name = config.model.lower()

    # Handle Siamese wrappers for instance-recognition
    if "instance-recognition" in config.dataset:
        if model_name == "mamba":
            return SiameseMamba(
                input_dim, output_dim, config.hidden_dimension, config.num_layers
            )
        elif model_name == "vae":
            return SiameseVAE(input_dim, output_dim, config.hidden_dimension)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]

    # Instantiate with common parameters (simplified for this factory)
    if model_name == "transformer":
        return Transformer(
            input_dim,
            output_dim,
            config.num_heads,
            config.hidden_dimension,
            config.num_layers,
            config.dropout,
        )
    elif model_name == "cnn":
        return CNN(input_dim, output_dim)
    elif model_name == "lstm":
        return LSTM(input_dim, output_dim, config.hidden_dimension, config.num_layers)
    elif model_name == "dense":
        return Dense(input_dim, output_dim, config.hidden_dimension)
    elif model_name == "moe":
        return MOE(input_dim, output_dim, config.hidden_dimension, config.num_layers)
    elif model_name == "ensemble":
        return Ensemble(input_dim, config.hidden_dimension, output_dim, config.dropout)

    # Fallback for models that might take (input_dim, output_dim)
    try:
        return model_class(input_dim, output_dim)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate model '{model_name}': {e}")

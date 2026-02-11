# -*- coding: utf-8 -*-
"""
Fishy business project package for spectral data analysis.
"""

from .engine.trainer import Trainer
from .engine.training_loops import train_model, evaluate_model
from .data.module import DataModule, create_data_module
from ._core.config import TrainingConfig
from ._core.factory import create_model

__all__ = [
    "Trainer",
    "train_model",
    "evaluate_model",
    "DataModule",
    "create_data_module",
    "TrainingConfig",
    "create_model",
]

# -*- coding: utf-8 -*-
"""
Fishy business project package for spectral data analysis.

Examples:
    >>> import fishy
    >>> hasattr(fishy, 'Trainer')
    True
"""

from .engine.trainer import Trainer, DeepEngine
from .data.module import DataModule, create_data_module
from ._core.config import TrainingConfig
from ._core.factory import create_model

__all__ = [
    "Trainer",
    "DeepEngine",
    "DataModule",
    "create_data_module",
    "TrainingConfig",
    "create_model",
]

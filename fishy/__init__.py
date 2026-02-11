# -*- coding: utf-8 -*-
"""
Fishy business project package for spectral data analysis.
"""

import warnings
import os

# 1. Suppress noisy library warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# Suppress some matplotlib chatter
os.environ['MPLBACKEND'] = 'Agg' 

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

# -*- coding: utf-8 -*-
"""
Fishy business project package for spectral data analysis.
"""

import warnings
import os

# 1. Suppress noisy library warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

# Silencing NotOpenSSLWarning (common on macOS with older system Python)
warnings.filterwarnings("ignore", message=".*OpenSSL 1.1.1+.*")
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# Suppress some matplotlib chatter
os.environ["MPLBACKEND"] = "Agg"

from .engine.trainer import Trainer, DeepEngine
from .data.module import DataModule, create_data_module
from ._core.config import TrainingConfig
from ._core.factory import create_model
from .experiments.unified_trainer import run_unified_training
from .cli.main import display_final_summary


def get_data_path(filename: str = "REIMS.xlsx") -> str:
    """Returns the absolute path to a data asset within the package."""
    import importlib.resources as pkg_resources
    
    try:
        # Modern way to get the resource path
        with pkg_resources.path("fishy.data.assets", filename) as p:
            if p.exists():
                return str(p)
    except (ImportError, FileNotFoundError, TypeError):
        pass

    # Fallback 1: Local development (within the package)
    local_p = os.path.join(os.path.dirname(__file__), "data", "assets", filename)
    if os.path.exists(local_p):
        return os.path.abspath(local_p)
    
    # Fallback 2: Project root data directory
    root_p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", filename))
    if os.path.exists(root_p):
        return root_p

    # Final fallback: Return the most likely path string even if it doesn't exist
    return root_p


__all__ = [
    "Trainer",
    "DeepEngine",
    "DataModule",
    "create_data_module",
    "TrainingConfig",
    "create_model",
    "run_unified_training",
    "display_final_summary",
    "get_data_path",
]

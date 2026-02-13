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
    """Returns the absolute path to a data asset, searching package and local dirs."""
    import importlib.resources as pkg_resources
    import os

    # 1. Determine package directory
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. List all potential search paths
    search_paths = [
        # Standard: Inside the installed package
        os.path.join(pkg_dir, "data", "assets", filename),
        # Local Dev: Root data folder relative to package
        os.path.abspath(os.path.join(pkg_dir, "..", "data", filename)),
        # RTD/CI: Current working directory's data folder
        os.path.join(os.getcwd(), "data", filename),
        # RTD/CI: Current working directory's internal assets folder
        os.path.join(os.getcwd(), "fishy", "data", "assets", filename),
    ]

    # 3. Return the first one that exists
    for p in search_paths:
        if os.path.exists(p):
            return p

    # 4. Fallback for download: Use the internal package location if writable, else CWD
    preferred_path = search_paths[0]
    try:
        os.makedirs(os.path.dirname(preferred_path), exist_ok=True)
        return preferred_path
    except (OSError, PermissionError):
        return os.path.join(os.getcwd(), "data", filename)


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

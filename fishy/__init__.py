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

    # 1. Try modern importlib.resources
    try:
        with pkg_resources.path("fishy.data.assets", filename) as p:
            if p.parent.exists(): # Check parent dir because file might not be downloaded yet
                return str(p)
    except (ImportError, FileNotFoundError, TypeError):
        pass

    # 2. Fallback to package relative path (robust across installs)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    asset_p = os.path.join(pkg_dir, "data", "assets", filename)
    if os.path.exists(os.path.dirname(asset_p)):
        return asset_p

    # 3. Last resort fallback to project root data directory (local dev only)
    root_p = os.path.abspath(
        os.path.join(pkg_dir, "..", "data", filename)
    )
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

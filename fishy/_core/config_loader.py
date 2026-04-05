# -*- coding: utf-8 -*-
"""
Utility for loading YAML configuration files.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml

_CONFIG_CACHE: Dict[str, Any] = {}


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the configs directory.
    Uses caching to avoid redundant file I/O.

    Args:
        config_name (str): Name of the YAML file (without .yaml extension).

    Returns:
        Dict[str, Any]: The configuration data.

    Examples:
        >>> datasets = load_config("datasets")
        >>> "species" in datasets
        True
        >>> models = load_config("models")
        >>> "deep_models" in models
        True
    """
    if config_name in _CONFIG_CACHE:
        return _CONFIG_CACHE[config_name]

    config_path = (
        Path(__file__).resolve().parent.parent / "configs" / f"{config_name}.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    _CONFIG_CACHE[config_name] = config_data
    return config_data


def detect_method(model_name: str) -> str:
    """Automatically detects the training method category for a given model name."""
    try:
        cfg = load_config("models")
        m = model_name.lower()
        if m in cfg.get("deep_models", {}):
            return "deep"
        if m in cfg.get("classic_models", {}):
            return "classic"
        if m in cfg.get("evolutionary_models", {}):
            return "evolutionary"
        if m in cfg.get("contrastive_models", {}):
            return "contrastive"
        if m in cfg.get("probabilistic_models", {}):
            return "probabilistic"
    except Exception:
        pass
    return "deep"

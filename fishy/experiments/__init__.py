# -*- coding: utf-8 -*-
"""
High-level experiment orchestrators and runners.
"""

from .classic_training import run_sklearn_experiment as run_classic_experiment
from .classic_training import run_sklearn_experiment as run_evolutionary_experiment
from .contrastive import run_contrastive_experiment
from .deep_training import run_training_pipeline
from .transfer import run_sequential_transfer_learning
from .unified_trainer import run_unified_training, run_all_benchmarks

__all__ = [
    "run_classic_experiment",
    "run_evolutionary_experiment",
    "run_training_pipeline",
    "run_sequential_transfer_learning",
    "run_sklearn_experiment",
    "run_unified_training",
    "run_all_benchmarks",
]

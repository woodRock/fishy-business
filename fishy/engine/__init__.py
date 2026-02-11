# -*- coding: utf-8 -*-
"""
Execution engine for training and evaluation.
"""

from .trainer import Trainer
from .training_loops import train_model, evaluate_model, transfer_learning
from .losses import coral_loss, cumulative_link_loss, levels_from_labelbatch

__all__ = [
    "Trainer",
    "train_model",
    "evaluate_model",
    "transfer_learning",
    "coral_loss",
    "cumulative_link_loss",
    "levels_from_labelbatch",
]

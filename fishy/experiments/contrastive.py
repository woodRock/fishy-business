# -*- coding: utf-8 -*-
"""
Contrastive learning experiments module.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

from fishy.data.module import create_data_module
from fishy.data.augmentation import AugmentationConfig, DataAugmenter
from fishy.data.contrastive_util import DataConfig, DataPreprocessor, SiameseDataset, BalancedBatchSampler
from fishy.models.contrastive.simclr import SimCLRModel, SimCLRLoss
from fishy.models.contrastive.moco import MoCoModel, MoCoLoss
from fishy.models.contrastive.byol import BYOLModel, BYOLLoss
from fishy.models.contrastive.simsiam import SimSiamModel, SimSiamLoss
from fishy.models.contrastive.barlow_twins import BarlowTwinsModel, BarlowTwinsLoss
from fishy._core.factory import create_model, MODEL_REGISTRY
from fishy._core.config import TrainingConfig

@dataclass
class ContrastiveConfig:
    num_runs: int = 1
    temperature: float = 0.55
    projection_dim: int = 256
    embedding_dim: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    batch_size: int = 16
    num_epochs: int = 100
    input_dim: int = 2080
    encoder_type: str = "transformer"
    contrastive_method: str = "simclr"

def run_contrastive_experiment(config: ContrastiveConfig):
    """
    Orchestrates contrastive learning experiments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # ... logic from contrastive/main.py adapted here ...
    pass

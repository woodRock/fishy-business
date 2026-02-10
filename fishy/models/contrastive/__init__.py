# -*- coding: utf-8 -*-
"""
Contrastive learning models.
"""

from .simclr import SimCLRModel, SimCLRLoss
from .moco import MoCoModel, MoCoLoss
from .byol import BYOLModel, BYOLLoss
from .simsiam import SimSiamModel, SimSiamLoss
from .barlow_twins import BarlowTwinsModel, BarlowTwinsLoss

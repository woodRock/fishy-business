# -*- coding: utf-8 -*-
"""
Deep learning models for spectral analysis.
"""

from .cnn import CNN
from .dense import Dense
from .ensemble import Ensemble
from .hybrid import Hybrid
from .kan import KAN
from .lstm import LSTM
from .mamba import Mamba, SiameseMamba
from .MOE import MOE
from .ode import ODE
from .ordinal import TransformerOrdinal
from .performer import Performer
from .rcnn import RCNN
from .rwkv import RWKV
from .tcn import TCN
from .transformer import Transformer
from .vae import VAE, SiameseVAE
from .wavenet import WaveNet

__all__ = [
    "CNN",
    "Dense",
    "Ensemble",
    "Hybrid",
    "KAN",
    "LSTM",
    "Mamba",
    "SiameseMamba",
    "MOE",
    "ODE",
    "TransformerOrdinal",
    "Performer",
    "RCNN",
    "RWKV",
    "TCN",
    "Transformer",
    "VAE",
    "SiameseVAE",
    "WaveNet",
]

"""Package initialization for the 'models' module.

This module initializes the 'models' package, allowing for easy import of various model architectures.
It includes implementations of different neural network architectures suitable for tasks such as
time series classification, image processing, and more.

"""

# models/__init__.py


# If you have submodules within 'models' that you want to expose
# when someone does 'import models', you can do it here.
# For example, if you had 'model/submodule.py':
# from .submodule import MyClass

from .cnn import CNN
from .dense import Dense
from .diffusion import Diffusion
from .ensemble import Ensemble
from .kan import KAN
from .lstm import LSTM
from .mamba import Mamba
from .MOE import MOE
from .ode import ODE
from .rcnn import RCNN
from .rwkv import RWKV
from .tcn import TCN
from .transformer import Transformer, MultiHeadAttention
from .vae import VAE
from .wavenet import WaveNet
from .hybrid import Hybrid

# from .longformer import Longformer
from .performer import Performer
from .moco import MoCoModel, MoCoLoss
from .byol import BYOLModel, BYOLLoss
from .simsiam import SimSiamModel, SimSiamLoss
from .barlow_twins import BarlowTwinsModel, BarlowTwinsLoss
from .simclr import SimCLRModel, SimCLRLoss
from .vae import SiameseVAE

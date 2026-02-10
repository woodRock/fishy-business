# -*- coding: utf-8 -*-
"""
Model factory for creating deep learning models.
"""

import torch
import torch.nn as nn
from typing import Dict, Type

from fishy.models.deep import (
    Transformer,
    LSTM,
    CNN,
    RCNN,
    Mamba,
    SiameseMamba,
    KAN,
    VAE,
    SiameseVAE,
    MOE,
    Dense,
    ODE,
    RWKV,
    TCN,
    WaveNet,
    Ensemble,
    Diffusion,
)
from fishy._core.config import TrainingConfig

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "transformer": Transformer,
    "lstm": LSTM,
    "cnn": CNN,
    "rcnn": RCNN,
    "mamba": Mamba,
    "kan": KAN,
    "vae": VAE,
    "moe": MOE,
    "dense": Dense,
    "ode": ODE,
    "rwkv": RWKV,
    "tcn": TCN,
    "wavenet": WaveNet,
    "ensemble": Ensemble,
    "diffusion": Diffusion,
}

class SiameseWrapper(nn.Module):
    def __init__(self, base_model, embedding_dim):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # 2 classes: same/different
        )

    def forward(self, x1, x2):
        emb1 = self.base_model(x1)
        emb2 = self.base_model(x2)
        diff = torch.abs(emb1 - emb2)
        return self.classifier(diff)

def create_model(config: TrainingConfig, input_dim: int, output_dim: int) -> nn.Module:
    """
    Creates a model instance based on the model specified in the config.
    """
    model_class = MODEL_REGISTRY.get(config.model)
    if not model_class:
        raise ValueError(f"Invalid model type: {config.model}")

    if config.use_coral or config.use_cumulative_link:
        output_dim = output_dim - 1

    model_args = {"dropout": config.dropout}

    if "instance-recognition" in config.dataset:
        embedding_dim = config.hidden_dimension

        if config.model == "mamba":
            mamba_model = Mamba(
                input_dim=input_dim,
                d_model=config.hidden_dimension,
                d_state=config.hidden_dimension,
                d_conv=4,
                expand=2,
                depth=config.num_layers,
            )
            return SiameseMamba(mamba_model)

        if config.model == "vae":
            vae_model = VAE(
                input_size=input_dim,
                num_classes=output_dim,
                latent_dim=config.hidden_dimension,
                **model_args,
            )
            return SiameseVAE(vae_model)

        if config.model == "transformer":
            base_model = Transformer(
                input_dim=input_dim,
                output_dim=embedding_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dimension,
                **model_args,
            )
        elif config.model == "ensemble":
            base_model = Ensemble(
                input_dim=input_dim,
                output_dim=embedding_dim,
                hidden_dim=config.hidden_dimension,
                dropout=config.dropout,
            )
        elif config.model == "lstm":
            base_model = LSTM(
                input_dim=input_dim,
                output_dim=embedding_dim,
                hidden_dim=config.hidden_dimension,
                num_layers=config.num_layers,
                **model_args,
            )
        elif config.model in ["cnn", "rcnn"]:
            base_model = model_class(
                input_dim=input_dim, output_dim=embedding_dim, **model_args
            )
        elif config.model == "kan":
            base_model = KAN(
                input_dim=input_dim,
                output_dim=embedding_dim,
                hidden_dim=config.hidden_dimension,
                num_layers=config.num_layers,
                dropout_rate=config.dropout,
                num_inner_functions=10,
            )
        elif config.model == "moe":
            base_model = MOE(
                input_dim=input_dim,
                output_dim=embedding_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                hidden_dim=config.hidden_dimension,
                num_experts=4,
                k=2,
            )
        else:
            base_model = model_class(
                input_dim=input_dim, output_dim=embedding_dim, **model_args
            )

        return SiameseWrapper(base_model, embedding_dim)

    if config.model == "transformer":
        model_args.update(
            {
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "hidden_dim": config.hidden_dimension,
            }
        )
        return Transformer(input_dim=input_dim, output_dim=output_dim, **model_args)
    elif config.model == "ensemble":
        return Ensemble(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dimension,
            dropout=config.dropout,
        )
    elif config.model == "lstm":
        model_args.update(
            {"hidden_dim": config.hidden_dimension, "num_layers": config.num_layers}
        )
        return LSTM(input_dim=input_dim, output_dim=output_dim, **model_args)
    elif config.model in ["cnn", "rcnn"]:
        return model_class(input_dim=input_dim, output_dim=output_dim, **model_args)
    elif config.model == "mamba":
        return Mamba(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=config.hidden_dimension,
            d_state=config.hidden_dimension,
            d_conv=4,
            expand=2,
            depth=config.num_layers,
        )
    elif config.model == "kan":
        return KAN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dimension,
            num_layers=config.num_layers,
            dropout_rate=config.dropout,
            num_inner_functions=10,
        )
    elif config.model == "vae":
        return VAE(
            input_size=input_dim,
            num_classes=output_dim,
            latent_dim=config.hidden_dimension,
            **model_args,
        )
    elif config.model == "moe":
        return MOE(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dimension,
            num_experts=4,
            k=2,
        )
    elif config.model == "diffusion":
        return Diffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dimension,
            time_dim=64,
            num_timesteps=1000,
        )
    else:
        return model_class(input_dim=input_dim, output_dim=output_dim, **model_args)

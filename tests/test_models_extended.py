# -*- coding: utf-8 -*-
import pytest
import torch
import numpy as np
from fishy._core.factory import create_model
from fishy._core.config import TrainingConfig
from fishy.models.deep.rwkv import RWKV
from fishy.models.deep.rcnn import RCNN
from fishy.models.deep.MOE import MOE
from fishy.models.classic.opls_da import OPLS_DA

def test_rwkv_instantiation_and_forward():
    input_dim, hidden_dim, output_dim = 100, 64, 5
    model = RWKV(input_dim, output_dim, hidden_dim=hidden_dim)
    x = torch.randn(8, input_dim)
    out = model(x)
    assert out.shape == (8, output_dim)

def test_rcnn_instantiation_and_forward():
    input_dim, output_dim = 100, 5
    model = RCNN(input_dim, output_dim, dropout=0.2)
    x = torch.randn(8, input_dim)
    out = model(x)
    assert out.shape == (8, output_dim)

def test_moe_instantiation_and_forward():
    input_dim, output_dim, hidden_dim, num_experts = 100, 5, 64, 4
    model = MOE(input_dim, output_dim, hidden_dim=hidden_dim, num_experts=num_experts)
    x = torch.randn(8, input_dim)
    out = model(x)
    assert out.shape == (8, output_dim)

def test_opls_da_classic():
    model = OPLS_DA(n_components=2)
    X = np.random.randn(20, 100)
    y = np.random.randint(0, 2, 20)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 20
    assert hasattr(model, "predict_proba")
    probs = model.predict_proba(X)
    assert probs.shape == (20, 2)

def test_factory_extended():
    # Test RWKV via factory
    cfg = TrainingConfig(model="rwkv", hidden_dimension=32)
    model = create_model(cfg, 100, 5)
    assert isinstance(model, RWKV)
    
    # Test MoE via factory
    cfg_moe = TrainingConfig(model="moe", hidden_dimension=32, num_layers=3)
    model_moe = create_model(cfg_moe, 100, 5)
    assert isinstance(model_moe, MOE)

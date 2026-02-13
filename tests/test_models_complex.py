# -*- coding: utf-8 -*-
import pytest
import torch
from fishy.models.deep.mamba import Mamba
from fishy.models.deep.ode import ODE
from fishy.models.deep.kan import KAN
from fishy.models.deep.tcn import TCN
from fishy.models.deep.wavenet import WaveNet
from fishy.models.deep.performer import Performer
from fishy.models.deep.hybrid import Hybrid


def test_mamba_forward():
    input_dim, output_dim = 100, 5
    model = Mamba(input_dim, output_dim, d_model=64, n_layers=2)
    x = torch.randn(8, input_dim)
    out = model(x)
    assert out.shape == (8, output_dim)


def test_ode_forward():
    input_dim, output_dim = 100, 5
    model = ODE(input_dim, output_dim, hidden_dim=64)
    x = torch.randn(8, input_dim)
    out = model(x)
    assert out.shape == (8, output_dim)


def test_kan_forward():
    input_dim, output_dim = 100, 5
    model = KAN(input_dim, output_dim, hidden_dim=64)
    x = torch.randn(8, input_dim)
    out = model(x)
    assert out.shape == (8, output_dim)


def test_tcn_forward():
    input_dim, output_dim = 100, 5
    model = TCN(input_dim, output_dim, num_channels=[32, 64])
    x = torch.randn(8, 1, input_dim)
    out = model(x)
    assert out.shape == (8, output_dim)


def test_wavenet_forward():
    input_dim, output_dim = 100, 5
    model = WaveNet(input_dim, output_dim, num_layers=2)
    x = torch.randn(8, 1, input_dim)
    out = model(x)
    assert out.shape == (8, output_dim)


def test_performer_forward():
    input_dim, output_dim = 100, 5
    model = Performer(input_dim, output_dim, hidden_dim=64, n_layers=2)
    x = torch.randn(8, input_dim)
    out = model(x)
    assert out.shape == (8, output_dim)


def test_hybrid_forward():
    input_dim, output_dim = 100, 5
    model = Hybrid(input_dim, output_dim, hidden_dim=64)
    x = torch.randn(8, input_dim)
    out = model(x)
    assert out.shape == (8, output_dim)

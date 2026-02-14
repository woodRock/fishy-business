# -*- coding: utf-8 -*-
"""
Unit tests for Sparsely-Gated Mixture of Experts (GMOE).
"""

import pytest
import torch
from fishy.models.deep.gmoe import SparselyGatedMoE

def test_gmoe_initialization():
    input_dim = 128
    output_dim = 2
    model = SparselyGatedMoE(input_dim=input_dim, output_dim=output_dim, num_experts=4, k=1)
    assert len(model.experts) == 4
    assert model.k == 1

def test_gmoe_forward_pass():
    batch_size = 8
    input_dim = 128
    output_dim = 3
    model = SparselyGatedMoE(input_dim=input_dim, output_dim=output_dim, num_experts=4, k=2)
    
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, output_dim)
    # Output should not be all zeros
    assert not torch.all(output == 0)

def test_gmoe_3d_input():
    batch_size = 4
    input_dim = 128
    output_dim = 2
    model = SparselyGatedMoE(input_dim=input_dim, output_dim=output_dim, num_experts=2, k=1)
    
    x = torch.randn(batch_size, 1, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, output_dim)

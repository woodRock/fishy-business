# -*- coding: utf-8 -*-
import pytest
import torch
import torch.nn as nn
import numpy as np
from fishy.analysis.xai import GradCAM, ModelWrapper
from fishy.analysis.benchmark import run_benchmark
from fishy._core.utils import RunContext


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 100, 2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_gradcam_logic():
    model = SimpleModel()
    target_layer = model.conv
    gc = GradCAM(model, target_layer)
    x = torch.randn(1, 1, 100)
    cam = gc.generate_cam(x)
    assert cam.shape[-1] == 100
    gc.remove_hooks()


def test_model_wrapper():
    model = SimpleModel()
    wrapper = ModelWrapper(model, device="cpu")
    X = np.random.randn(5, 100).astype(np.float32)
    probs = wrapper.predict_proba(X)
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_benchmark_utility(tmp_path):
    model = SimpleModel()
    ctx = RunContext("test", "test", "test")
    # Mocking basic benchmark run
    run_benchmark(model, input_dim=100, device="cpu", ctx=ctx, training_time=1.0)
    # Search recursively for the figure
    matches = list(ctx.run_dir.glob("**/latency_distribution.png"))
    assert len(matches) > 0 or (ctx.run_dir / "latency_distribution.png").exists()

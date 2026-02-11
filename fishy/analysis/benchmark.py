# -*- coding: utf-8 -*-
"""
Benchmarking utility for measuring model performance.
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging

from fishy._core.utils import NumpyEncoder

logger = logging.getLogger(__name__)

def measure_model_size(model: torch.nn.Module) -> float:
    """
    Returns model size in MB.

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 10)
        >>> size = measure_model_size(model)
        >>> size > 0
        True
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def measure_inference_performance(
    model: torch.nn.Module, 
    input_size: tuple, 
    device: torch.device, 
    num_iterations: int = 100
) -> Dict[str, float]:
    """Measures inference latency and throughput."""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    latency = (total_time / num_iterations) * 1000  # ms
    throughput = (num_iterations * input_size[0]) / total_time  # samples/s
    
    return {"latency_ms": latency, "throughput_samples_per_s": throughput}

def get_peak_vram() -> float:
    """
    Returns peak VRAM usage in MB.

    Examples:
        >>> size = get_peak_vram()
        >>> isinstance(size, float)
        True
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0

def run_benchmark(
    model: Any, 
    input_dim: int, 
    device: torch.device, 
    ctx: Any,
    training_time: Optional[float] = None
) -> Dict[str, Any]:
    """Runs a full benchmark suite and saves results."""
    logger.info("Running performance benchmark...")
    results = {}
    
    if isinstance(model, torch.nn.Module):
        results["model_size_mb"] = measure_model_size(model)
        perf = measure_inference_performance(model, (1, input_dim), device)
        results.update(perf)
        results["peak_vram_mb"] = get_peak_vram()
        
        # Simple FLOPs estimation (very rough for linear layers)
        flops = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                flops += module.in_features * module.out_features
        results["estimated_flops_linear"] = flops
    else:
        # For non-torch models (classic/evolutionary)
        results["model_type"] = str(type(model))
        # Simple latency test if predict exists
        try:
            dummy_input = np.random.randn(1, input_dim)
            start = time.time()
            for _ in range(100):
                _ = model.predict(dummy_input)
            results["latency_ms"] = ((time.time() - start) / 100) * 1000
        except:
            pass

    if training_time:
        results["training_time_s"] = training_time

    # Save to benchmark folder
    benchmark_path = ctx.benchmark_dir / "performance.json"
    with open(benchmark_path, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    logger.info(f"Benchmark results saved to {benchmark_path}")
    
    if ctx.wandb_run:
        ctx.wandb_run.log({"benchmark": results}, commit=False)
        
    return results

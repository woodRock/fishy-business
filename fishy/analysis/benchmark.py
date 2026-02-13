# -*- coding: utf-8 -*-
"""
Benchmarking utilities for model performance evaluation.
"""

import time
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def measure_model_size(model: torch.nn.Module) -> float:
    """Calculates the model size in Megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def get_peak_vram() -> float:
    """Gets peak VRAM usage in MB (if CUDA is available)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def measure_inference_performance(
    model: torch.nn.Module,
    input_size: tuple,
    device: Union[torch.device, str],
    num_iterations: int = 100,
) -> Dict[str, float]:
    """Measures inference latency and throughput."""
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    latencies = []
    if device.type == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        with torch.no_grad():
            for _ in range(num_iterations):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))
    else:
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(dummy_input)
                latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = np.mean(latencies)
    throughput = (1 / (avg_latency / 1000)) if avg_latency > 0 else 0
    return {
        "avg_inference_latency_ms": float(avg_latency),
        "std_inference_latency_ms": float(np.std(latencies)),
        "throughput_samples_per_s": float(throughput),
    }


def run_benchmark(
    model: Any,
    input_dim: int,
    device: torch.device,
    ctx: Any,
    training_time: Optional[float] = None,
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
            X_dummy = np.random.randn(1, input_dim)
            start = time.perf_counter()
            for _ in range(10):
                _ = model.predict(X_dummy)
            results["avg_inference_latency_ms"] = (
                (time.perf_counter() - start) / 10
            ) * 1000
        except:
            pass

    if training_time:
        results["training_time_s"] = training_time

    # Save results
    ctx.save_results(results, filename="benchmark_results.json")

    # Generate Latency Distribution Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    if "avg_inference_latency_ms" in results:
        plt.bar(["Latency (ms)"], [results["avg_inference_latency_ms"]])
        plt.title("Inference Latency")
        ctx.save_figure(plt, "latency_distribution.png")
    plt.close()

    return results

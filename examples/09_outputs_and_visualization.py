# -*- coding: utf-8 -*-
"""
Tutorial 09: Outputs and Visualization
--------------------------------------
This tutorial explains where to find the results of your experiments
and how to interpret the generated artifacts.
"""

from fishy._core.utils import RunContext
from pathlib import Path


def main():
    print("--- Tutorial 09: Outputs and Visualization ---")

    # Every time you run an experiment, a `RunContext` is created.
    # It automatically creates a structured output directory:
    # outputs/{dataset}/{method}/{model}_{timestamp}/

    ctx = RunContext(dataset="species", method="deep", model_name="transformer")

    print(f"\nRun directory created at: {ctx.run_dir}")

    # 1. Logs: Found in {run_dir}/logs/experiment.log
    print(f"  Logs:       {ctx.log_dir}")

    # 2. Metrics: Found in {run_dir}/results/metrics.json
    # These are saved using the custom NumpyEncoder.
    print(f"  Results:    {ctx.result_dir}")

    # 3. Figures: Found in {run_dir}/figures/
    # If `figures=True` is set in config, you'll see training curves here.
    print(f"  Figures:    {ctx.figure_dir}")

    # 4. Checkpoints: Found in {run_dir}/checkpoints/
    # Best model weights are saved here during training.
    print(f"  Checkpoints: {ctx.checkpoint_dir}")

    print("\nTo generate these automatically during a run, ensure your config has:")
    print("  config.benchmark = True")
    print("  config.figures = True")


if __name__ == "__main__":
    main()

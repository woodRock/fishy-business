# -*- coding: utf-8 -*-
"""
Tutorial 05: Automated Benchmarking (Run All)
---------------------------------------------
This tutorial shows how to trigger the full automated benchmark suite
used in the paper, covering all datasets and model categories.
"""

from fishy.experiments.unified_trainer import run_all_benchmarks


def main():
    print("--- Tutorial 05: Automated Benchmarking ---")

    # The `run_all_benchmarks` function is a "one-button" solution to
    # compare every Classic, Deep, and Evolutionary model against each other.
    # It performs repeated cross-validation and statistical significance tests.

    # For this example, we use `quick=True` to run a very small subset
    # (2 models, 1 dataset, 2 runs) instead of the full 30-run suite.
    print("Launching quick benchmark suite...")

    summary_df = run_all_benchmarks(quick=True, wandb_log=False)

    print("\nBenchmark Summary Table:")
    print(summary_df.to_string())


if __name__ == "__main__":
    main()

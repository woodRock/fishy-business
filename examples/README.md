# Fishy Business: Tutorial Series

This folder contains a step-by-step tutorial on how to use the `fishy-business` codebase for mass spectrometry analysis.

### Tutorial Order

1.  **[Getting Started](01_getting_started.py)**: The simplest way to run a training experiment.
2.  **[DataModule](02_data_module.py)**: Loading, filtering, and inspecting datasets.
3.  **[Configuration](03_configuration.py)**: Managing hyperparameters with `TrainingConfig` and `ExperimentConfig`.
4.  **[Training Engines](04_training_engines.py)**: High-level orchestration vs. low-level custom loops.
5.  **[Automated Benchmarking](05_run_all_benchmarks.py)**: Running the full "Run All" suite.
6.  **[Self-Supervised Pre-training](06_pretraining.py)**: Leveraging unlabeled data.
7.  **[Transfer Learning](07_transfer_learning.py)**: Sequential knowledge transfer across datasets.
8.  **[Probabilistic Inference](08_probabilistic_inference.py)**: Bayesian models and uncertainty.
9.  **[Outputs and Visualization](09_outputs_and_visualization.py)**: Finding and interpreting results.

---

### Running the Tutorials

You can run any of these tutorials from the root of the repository:

```bash
python3 examples/01_getting_started.py
```

Most tutorials use the internal REIMS data file included with the package. You don't need to manually provide a data file unless you are using custom datasets.

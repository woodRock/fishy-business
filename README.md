# 🐟 Fishy Business
## Machine Learning for Rapid Evaporative Ionization Mass Spectrometry

**A Doctoral Thesis by Jesse Wood**
*Victoria University of Wellington*

[![Documentation Status](https://readthedocs.org/projects/fishy-business/badge/?version=latest)](https://fishy-business.readthedocs.io/en/latest/?badge=latest)
[![Format Python Code](https://github.com/woodRock/fishy-business/actions/workflows/black.yml/badge.svg)](https://github.com/woodRock/fishy-business/actions/workflows/black.yml)
[![Unit Tests](https://github.com/woodRock/fishy-business/actions/workflows/unittests.yml/badge.svg)](https://github.com/woodRock/fishy-business/actions/workflows/unittests.yml)
[![Coverage Status](https://coveralls.io/repos/github/woodRock/fishy-business/badge.svg?branch=main)](https://coveralls.io/github/woodRock/fishy-business?branch=main)
[![Doctests](https://github.com/woodRock/fishy-business/actions/workflows/doctests.yml/badge.svg)](https://github.com/woodRock/fishy-business/actions/workflows/doctests.yml)

A configuration-driven framework for the analysis of spectral data using Deep Learning, Classic Machine Learning, and Evolutionary Algorithms.

## Key Features

- **Configuration-Driven Architecture**: Add new datasets, models, or tasks by simply editing YAML files in `fishy/configs/`.
- **Unified Training Engine**: Centralized `Trainer` class handles loops, metrics, and early stopping consistently across all experiments.
- **Self-Supervised Learning**: Modular `PreTrainingOrchestrator` with support for 7+ pretext tasks (Masked Spectra, Denoising, etc.).
- **Contrastive Suite**: Implementation of SimCLR, SimSiam, BYOL, Barlow Twins, and MoCo.
- **Advanced Workflows**: Sequential transfer learning and Genetic Programming (GP) experiments.
- **Automated Verification**: Integrated `doctests` ensure documentation examples always stay functional.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/woodRock/fishy-business.git
    cd fishy-business
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The framework provides a unified CLI via `main.py`.

### 🪄 Getting Started (Recommended)
New users should start with the **Interactive Wizard**, which guides you through model selection, dataset choice, and analysis setup:
```bash
python3 main.py wizard
```
The wizard will generate the exact CLI command or a configuration file for you.

### 🌐 Interactive Dashboard
For visual data exploration, real-time training monitoring, and advanced biomarker analysis, use the built-in Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```
The dashboard provides interactive PCA/t-SNE clusters, spectral signature comparison, and deep-dive interpretability tools.

### 🚂 Training & Analysis
All model types (Deep, Classic, Evolutionary, Contrastive) are trained using the same unified command:

```bash
# Train a Deep Learning model with performance benchmarking
python3 main.py train -m transformer -d species --benchmark --figures

# Train a Classic ML model
python3 main.py train -m rf -d oil

# Train an Evolutionary Algorithm (Feature Weighting)
python3 main.py train -m ga -d part
```

### 📂 Config-Driven Experiments
For large-scale or reproducible experiments, use YAML configuration files:
```bash
python3 main.py train -c fishy/configs/experiments/quick_benchmark.yaml
```

### 🔬 Advanced Tasks
Expert flags are hidden by default to keep the interface clean. View them using:
```bash
# See context-aware help for transfer learning
python3 main.py train --transfer --help

# See ALL expert overrides (Hyperparameters, XAI, etc.)
python3 main.py train --all --help
```

### 📊 Full Benchmarking Suite
Run the full doctoral benchmarking suite with statistical analysis (paired t-tests):
```bash
python3 main.py run_all --num-runs 30
```

## Extending the Framework

The library is designed to be extended without modifying core logic:
- **New Dataset**: Add an entry to `fishy/configs/datasets.yaml` with filtering rules and label encoding type.
- **New Model**: Add the class path and default hyperparameters to `fishy/configs/models.yaml`.
- **New Pre-training Task**: Define the method and hyperparameters in `fishy/configs/pre_training.yaml`.

## Programmatic Usage

For advanced usage in Python scripts, see the `examples/` directory for a step-by-step tutorial series:
- `01_getting_started.py`: The simplest way to run a training experiment.
- `02_data_module.py`: Loading, filtering, and inspecting datasets.
- ... and 7 more tutorials covering pre-training, transfer learning, and more.

## Testing & Documentation

We maintain high code quality through automated testing:

```bash
# Run unit tests with coverage
pytest tests/

# Run documentation tests
pytest --doctest-modules fishy/
```

Comprehensive documentation is available at [Read the Docs](https://fishy-business.readthedocs.io/en/latest/).

## License

This project is licensed under the terms provided in the repository.

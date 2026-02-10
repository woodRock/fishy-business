# Machine Learning for Rapid Evaporative Ionization Mass Spectrometry for Marine Biomass Analysis
## A Doctoral Thesis by Jesse Wood

[![Documentation Status](https://readthedocs.org/projects/fishy-business/badge/?version=latest)](https://fishy-business.readthedocs.io/en/latest/?badge=latest)
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

### Standard Training (Classification)
```bash
python3 main.py train --model transformer --dataset species
```

### Self-Supervised Pre-training
```bash
python3 main.py pretrain --model mamba --masked-spectra-modelling --next-peak-prediction
```

### Ordinal & Standard Regression
```bash
python3 main.py ordinal --model lstm --dataset oil --use-coral
```

### Contrastive Learning
```bash
python3 main.py contrastive --method simclr --encoder transformer
```

### Benchmarking & Statistical Analysis
Run the full suite with paired t-tests against OPLS-DA baselines:
```bash
python3 main.py run_all --num-runs 30
```

## Extending the Framework

The library is designed to be extended without modifying core logic:
- **New Dataset**: Add an entry to `fishy/configs/datasets.yaml` with filtering rules and label encoding type.
- **New Model**: Add the class path and default hyperparameters to `fishy/configs/models.yaml`.
- **New Pre-training Task**: Define the method and hyperparameters in `fishy/configs/pre_training.yaml`.

## Programmatic Usage

For advanced usage in Python scripts, see the `examples/` directory:
- `01_programmatic_training.py`: High-level orchestration.
- `02_data_module_exploration.py`: Dynamic data loading.
- `03_low_level_trainer.py`: Custom PyTorch model training.

## Testing & Documentation

Run the live documentation tests:
```bash
pytest --doctest-modules fishy/
```

Comprehensive documentation is available at [Read the Docs](https://fishy-business.readthedocs.io/en/latest/).

## License

This project is licensed under the terms provided in the repository.

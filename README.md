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

> [!IMPORTANT]
> **Code vs. Data**: The source code in this repository is open-source under the MIT license. However, the accompanying REIMS dataset is private research data. You must use the `fishy download-data` command with an authorized token to fetch the data locally.

## Key Features

- **Consolidated CLI**: Run training, benchmarks, and the dashboard via a single `fishy` command.
- **Configuration-Driven Architecture**: Add new datasets, models, or tasks by simply editing YAML files in `fishy/configs/`.
- **Unified Training Engine**: Centralized `Trainer` class handles loops, metrics, and early stopping consistently across all experiments.
- **Advanced XAI Pipeline**: Automated biomarker discovery with direct mapping to chemical databases (LipidMaps).
- **Pro-Tier Architectures**: High-capacity Sparsely-Gated Mixture of Experts (`gmoe`) for complex spectral profiles.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/woodRock/fishy-business.git
    cd fishy-business
    ```

2.  **Install dependencies and CLI:**
    ```bash
    pip install -e .
    ```

## Usage

The framework provides a unified CLI via the `fishy` command.

### 🪄 Getting Started (Recommended)
New users should start with the **Interactive Wizard**, which guides you through model selection, dataset choice, and analysis setup:
```bash
fishy wizard
```

### 🌐 Interactive Dashboard
For visual data exploration, real-time training monitoring, and advanced biomarker analysis, use the built-in dashboard:
```bash
fishy dashboard
```

### 🚂 Training & Analysis
All model types (Deep, Classic, Evolutionary, Contrastive) are trained using the same unified command:

```bash
# Train a Gated MoE model with XAI biomarker discovery
fishy train -m gmoe -d species --xai

# Train a Deep Learning model with performance benchmarking
fishy train -m transformer -d species --benchmark --figures

# Train a Classic ML model
fishy train -m rf -d oil

# Train an Evolutionary Algorithm (Feature Weighting)
fishy train -m ga -d part
```

### 📂 Config-Driven Experiments
For large-scale or reproducible experiments, use YAML configuration files:
```bash
fishy train -c fishy/configs/experiments/quick_benchmark.yaml
```

### 🔬 Advanced Tasks
Expert flags are hidden by default to keep the interface clean. View them using:
```bash
# See context-aware help for transfer learning
fishy train --transfer --help

# See ALL expert overrides (Hyperparameters, XAI, etc.)
fishy train --all --help
```

### 📊 Full Benchmarking Suite
Run the full doctoral benchmarking suite with statistical analysis (paired t-tests):
```bash
fishy run_all --num-runs 30
```

## Extending the Framework

The library is designed to be extended without modifying core logic:
- **New Dataset**: Add an entry to `fishy/configs/datasets.yaml` with filtering rules and label encoding type.
- **New Model**: Add the class path and default hyperparameters to `fishy/configs/models.yaml`.
- **New Pre-training Task**: Define the method and hyperparameters in `fishy/configs/pre_training.yaml`.

## Programmatic Usage

For advanced usage in Python scripts, you can explore our tutorials in two ways:

### 📓 Interactive Tutorials (Recommended)
We provide Jupyter notebooks in the `notebooks/` directory matching the thesis chapters:
- **[01_Datasets and Preprocessing](notebooks/01_Datasets_and_Preprocessing.ipynb)**
- **[02_Species and Part Identification](notebooks/02_Species_and_Part_Identification.ipynb)**
- **[03_Oil and Cross-species Adulteration](notebooks/03_Oil_and_CrossSpecies_Adulteration.ipynb)**
- **[04_Contrastive Learning for Batch Detection](notebooks/04_Contrastive_Learning_for_Batch_Detection.ipynb)**

These are also rendered beautifully in our [online documentation](https://fishy-business.readthedocs.io/en/latest/tutorials.html).

## Docker

You can run the entire framework, including the dashboard, in a containerized environment:

1. **Build the image**:
   ```bash
   docker build -t fishy-business .
   ```

2. **Run the CLI**:
   ```bash
   docker run fishy-business fishy train -m transformer -d species
   ```

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite the following paper:

```bibtex
@article{wood2025hook,
  title={Hook, line, and spectra: machine learning for fish species identification and body part classification using rapid evaporative ionization mass spectrometry},
  author={Wood, Jesse and Nguyen, Bach and Xue, Bing and Zhang, Mengjie and Killeen, Daniel},
  journal={Intelligent Marine Technology and Systems},
  volume={3},
  number={1},
  pages={16},
  year={2025},
  publisher={Springer}
}
```

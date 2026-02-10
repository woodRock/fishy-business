# Machine Learning for Rapid Evaporative Ionization Mass Spectrometry for Marine Biomass Analysis
## A Doctoral Thesis by Jesse Wood

[![Documentation Status](https://readthedocs.org/projects/fishy-business/badge/?version=latest)](https://fishy-business.readthedocs.io/en/latest/?badge=latest)
[![Pylint](https://github.com/woodRock/fishy-business/actions/workflows/pylint.yml/badge.svg)](https://github.com/woodRock/fishy-business/actions/workflows/pylint.yml)

This repository contains the source code and documentation for my PhD research on spectral data analysis for fish species and oil classification.

## Getting Started

### Prerequisites

- Python 3.9 or higher.
- `pip` (Python package installer).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/woodRock/fishy-business.git
    cd fishy-business
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

## Usage

The project features a unified CLI entry point via `main.py`. You can run experiments, benchmarks, and analysis using subcommands.

### Run Training Pipeline
Train a deep learning model on a specific dataset:
```bash
python3 main.py train --model transformer --dataset species --epochs 50
```

### Benchmark Models
Compare performance across multiple architectures:
```bash
python3 main.py benchmark transformer cnn mamba moe
```

### Sequential Transfer Learning
Transfer weights across datasets:
```bash
python3 main.py transfer --model transformer --transfer-datasets oil --target-dataset part
```

### Model Explainability (XAI)
Generate LIME or Grad-CAM visualizations:
```bash
python3 main.py xai --method gradcam --dataset part --instance frames
```

### Evolutionary Experiments
Run Genetic Programming classification:
```bash
python3 main.py evolutionary --dataset species --generations 20
```

For more options and help, run:
```bash
python3 main.py --help
```

## Repository Organization

```
.
├── fishy/                  # Main research package
│   ├── _core/              # Core configuration and model factory
│   ├── analysis/           # Explainability tools (LIME, Grad-CAM)
│   ├── cli/                # Command-line interface implementation
│   ├── data/               # Data loading, processing, and augmentation
│   ├── engine/             # Execution engine (Training loops, losses)
│   ├── experiments/        # High-level experiment orchestrators
│   └── models/             # Model architectures (Deep, Classic, Evo)
├── main.py                 # Unified CLI entry point
├── data/                   # Raw data files (e.g., REIMS.xlsx)
├── docs/                   # Documentation and PhD notes
├── outputs/                # Artifacts from runs (figures, logs, checkpoints)
├── papers/                 # Free PDF versions of my published papers
└── requirements.txt        # Python dependencies
```

## Documentation

Comprehensive project documentation is available at [Read the Docs](https://fishy-business.readthedocs.io/en/latest/).

## License

This project is licensed under the terms provided in the repository.
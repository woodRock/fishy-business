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

- **Configuration-Driven Architecture**: Add new datasets, models, or tasks by simply editing YAML files in `fishy/configs/`.
- **Unified Training Engine**: Centralized `Trainer` class handles loops, metrics, and early stopping consistently across all experiments.
- **Self-Supervised Learning**: Modular `PreTrainingOrchestrator` with support for 7+ pretext tasks (Masked Spectra, Denoising, etc.).
- **Contrastive Suite**: Implementation of SimCLR, SimSiam, BYOL, Barlow Twins, and MoCo.
- **Advanced Workflows**: Sequential transfer learning and Genetic Programming (GP) experiments.
- **Automated Verification**: Integrated `doctests` ensure documentation examples always stay functional.

## Datasets & REIMS Data

This research utilizes high-dimensional chemical fingerprints (2,080 features) derived from **Rapid Evaporative Ionization Mass Spectrometry (REIMS)** of seafood samples, provided by **AgResearch, New Zealand**.

The data was collected using a Laser-Assisted REIMS setup in negative ionization mode, specifically targeting deprotonated lipids and fatty acids. The framework includes five distinct analytical tasks:

1.  **Species Identification**: Distinguishing between Hoki and Mackerel.
2.  **Body Part Identification**: Classifying 7 fish parts (fillets, heads, livers, skins, gonads, guts, frames).
3.  **Oil Contamination Detection**: Detecting contamination levels from 0.1% to 50%.
4.  **Cross-species Adulteration**: Identifying premixed samples of premium and cheaper species.
5.  **Batch Detection**: A pairwise task to identify if two fish belong to the same processing batch.

For full details on the acquisition and curation process, see the [Documentation](https://fishy-business.readthedocs.io/en/latest/datasets.html).

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

For advanced usage in Python scripts, you can explore our tutorials in two ways:

### 📓 Interactive Tutorials (Recommended)
We provide Jupyter notebooks in the `notebooks/` directory matching the thesis chapters:
- **[01_Datasets and Preprocessing](notebooks/01_Datasets_and_Preprocessing.ipynb)**
- **[02_Species and Part Identification](notebooks/02_Species_and_Part_Identification.ipynb)**
- **[03_Oil and Cross-species Adulteration](notebooks/03_Oil_and_CrossSpecies_Adulteration.ipynb)**
- **[04_Contrastive Learning for Batch Detection](notebooks/04_Contrastive_Learning_for_Batch_Detection.ipynb)**

These are also rendered beautifully in our [online documentation](https://fishy-business.readthedocs.io/en/latest/tutorials.html).

### 🐍 Python Examples
A step-by-step tutorial series is available in the `examples/` directory:
- `01_getting_started.py`: The simplest way to run a training experiment.
- `02_data_module.py`: Loading, filtering, and inspecting datasets.
- ... and 7 more tutorials covering pre-training, transfer learning, and more.

## Docker

You can run the entire framework, including the dashboard, in a containerized environment:

1. **Build the image**:
   ```bash
   docker build -t fishy-business .
   ```

2. **Run the dashboard**:
   ```bash
   docker run -p 8501:8501 fishy-business
   ```

3. **Persistence**: To save results locally, mount the output directories:
   ```bash
   docker run -p 8501:8501 -v $(pwd)/outputs:/app/outputs -v $(pwd)/logs:/app/logs fishy-business
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

For a full list of related research and publications, see the author's [Google Scholar](https://scholar.google.com/citations?hl=en&user=UnUp2S0AAAAJ) page.


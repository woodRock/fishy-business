# -*- coding: utf-8 -*-
"""
Tutorial 02: DataModule and Data Processing
-------------------------------------------
This tutorial explains how the `DataModule` handles data loading,
filtering, and conversion into PyTorch-ready tensors.
"""

from pathlib import Path
from fishy.data.module import create_data_module

# Path to the dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    print("--- Tutorial 02: DataModule and Data Processing ---")

    # 1. Create a DataModule
    # You can select different datasets defined in fishy/configs/datasets.yaml
    dataset_name = "species"
    dm = create_data_module(dataset_name=dataset_name)

    print(f"Initializing DataModule for: {dataset_name}")

    # 2. Setup the module
    # This triggers the actual loading from Excel/CSV and applies filters.
    dm.setup()

    # 3. Inspect metadata
    # The module automatically determines input dimension and classes from the data.
    print(f"  Input Dimension (features): {dm.get_input_dim()}")
    print(f"  Number of Classes:          {dm.get_num_classes()}")
    print(f"  Class Names:                {dm.get_class_names()}")

    # 4. Accessing Tensors
    # You can get the full dataset as NumPy arrays for inspection or traditional ML.
    X, y = dm.get_numpy_data(labels_as_indices=True)
    print(f"\nNumPy Data Shape: X={X.shape}, y={y.shape}")

    # 5. Accessing the PyTorch DataLoader
    # This is what's used during the deep learning training loop.
    loader = dm.get_train_dataloader()
    first_batch = next(iter(loader))
    spectra, labels = first_batch

    print("\nFirst PyTorch Batch:")
    print(f"  Spectra tensor shape: {spectra.shape}")
    print(f"  Labels tensor shape:  {labels.shape}")


if __name__ == "__main__":
    main()

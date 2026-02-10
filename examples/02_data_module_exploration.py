# -*- coding: utf-8 -*-
"""
Example 02: DataModule Exploration
----------------------------------
This script shows how to use the DataModule to load, filter, and inspect 
spectral data programmatically.
"""

from pathlib import Path
from fishy.data.module import create_data_module

# Set up the path to your data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")

def main():
    # 1. Create and setup the DataModule for a specific dataset
    dataset_name = "part" # Try 'species', 'oil', etc.
    data_module = create_data_module(
        dataset_name=dataset_name,
        file_path=DATA_PATH,
        batch_size=32
    )
    
    print(f"Loading dataset: {dataset_name}")
    data_module.setup()

    # 2. Inspect metadata determined from configs/datasets.yaml
    input_dim = data_module.get_input_dim()
    num_classes = data_module.get_num_classes()
    class_names = data_module.get_class_names()

    print(f"  Input Features: {input_dim}")
    print(f"  Number of Classes: {num_classes}")
    print(f"  Class Names: {class_names}")

    # 3. Access the raw underlying dataframe
    df = data_module.get_train_dataframe()
    print(f"
Raw DataFrame Shape: {df.shape}")
    print("Columns (First 5):", df.columns[:5].tolist())

    # 4. Access the PyTorch DataLoader
    loader = data_module.get_train_dataloader()
    first_batch = next(iter(loader))
    spectra, labels = first_batch
    
    print(f"
Batch Tensor Shapes:")
    print(f"  Spectra: {spectra.shape}")
    print(f"  Labels:  {labels.shape}")

if __name__ == "__main__":
    main()

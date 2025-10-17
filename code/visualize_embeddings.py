import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import argparse
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import sys

# Add the project root to the python path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from contrastive.main import (
    ContrastiveConfig,
    create_contrastive_model,
    visualize_batch_thresholds,
    ContrastiveTrainer,
)
from contrastive.util import (
    DataConfig,
    DataPreprocessor,
    SiameseDataset,
    BalancedBatchSampler,
)


def main(encoder_type, contrastive_method, model_path):
    """
    Generates and saves visualizations of learned embeddings for a trained contrastive model.
    """
    # 1. Configuration
    config = ContrastiveConfig(
        encoder_type=encoder_type,
        contrastive_method=contrastive_method,
        batch_size=16,
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 2. Data Loading and Preparation
    data_path = "/Users/woodj/Desktop/fishy-business/data/REIMS.xlsx"
    if not os.path.exists(data_path):
        data_path = "/vol/ecrg-solar/woodj4/fishy-business/data/REIMS.xlsx"

    data_config = DataConfig(batch_size=config.batch_size, data_path=data_path)
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data(data_config)
    filtered_data = preprocessor.filter_data(data, data_config.dataset_name)
    features = filtered_data.drop("m/z", axis=1).to_numpy()
    labels = preprocessor.encode_labels(filtered_data, data_config.dataset_name)

    full_dataset = SiameseDataset(features, labels)

    pair_indices = np.arange(len(full_dataset))
    pair_labels_for_stratify = np.argmax(full_dataset.pair_labels, axis=1)

    train_val_indices, test_indices = train_test_split(
        pair_indices, test_size=0.2, random_state=42, stratify=pair_labels_for_stratify
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.25,
        random_state=42,
        stratify=pair_labels_for_stratify[train_val_indices],
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_pair_labels = full_dataset.pair_labels[train_indices]
    val_pair_labels = full_dataset.pair_labels[val_indices]
    test_pair_labels = full_dataset.pair_labels[test_indices]

    train_sampler = BalancedBatchSampler(train_pair_labels, config.batch_size)
    val_sampler = BalancedBatchSampler(val_pair_labels, config.batch_size)
    test_sampler = BalancedBatchSampler(test_pair_labels, config.batch_size)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    print(
        f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test pairs."
    )

    # 3. Model Initialization and Loading
    model, _ = create_contrastive_model(config)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")
    else:
        print(
            f"Warning: Model path not found at {model_path}. Using randomly initialized model."
        )
    model.to(device)
    model.eval()

    # 4. Determine the Best Threshold from Training Data
    h1_list, h2_list, labels_list = [], [], []
    with torch.no_grad():
        for x1, x2, labels in train_loader:
            h1, h2 = model(x1.float().to(device), x2.float().to(device))
            h1_list.append(h1.detach())
            h2_list.append(h2.detach())
            labels_list.append(labels)

    if not h1_list:
        print("Training loader is empty, cannot determine threshold or generate plots.")
        return

    h1_cat = torch.cat(h1_list)
    h2_cat = torch.cat(h2_list)
    labels_cat = torch.cat(labels_list)
    best_threshold = ContrastiveTrainer._find_best_threshold(h1_cat, h2_cat, labels_cat)
    print(f"Best threshold found: {best_threshold:.4f}")

    # 5. Generate and Save Visualizations
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating visualization for Training set...")
    visualize_batch_thresholds(
        model,
        train_loader,
        device,
        f"Training Set ({contrastive_method.upper()}-{encoder_type.upper()})",
        os.path.join(
            output_dir, f"{contrastive_method}_{encoder_type}_train_embeddings.png"
        ),
        best_threshold,
    )

    print("Generating visualization for Validation set...")
    visualize_batch_thresholds(
        model,
        val_loader,
        device,
        f"Validation Set ({contrastive_method.upper()}-{encoder_type.upper()})",
        os.path.join(
            output_dir, f"{contrastive_method}_{encoder_type}_val_embeddings.png"
        ),
        best_threshold,
    )

    print("Generating visualization for Test set...")
    visualize_batch_thresholds(
        model,
        test_loader,
        device,
        f"Test Set ({contrastive_method.upper()}-{encoder_type.upper()})",
        os.path.join(
            output_dir, f"{contrastive_method}_{encoder_type}_test_embeddings.png"
        ),
        best_threshold,
    )

    print(f"Visualizations saved to '{output_dir}/' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize learned embeddings from a contrastive model."
    )
    parser.add_argument(
        "--encoder_type", type=str, default="cnn", help="Encoder architecture to use."
    )
    parser.add_argument(
        "--contrastive_method",
        type=str,
        default="simclr",
        help="Contrastive learning method used.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model_simclr_cnn_overall.pth",
        help="Path to the trained model weights.",
    )
    args = parser.parse_args()
    main(args.encoder_type, args.contrastive_method, args.model_path)

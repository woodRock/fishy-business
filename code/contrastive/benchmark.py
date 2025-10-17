
import argparse
import time
import torch
import numpy as np
import pandas as pd
import os
from contrastive.main import create_backbone_encoder, ContrastiveConfig
from contrastive.util import DataPreprocessor, SiameseDataset, DataConfig
from torch.utils.data import DataLoader

# Configure PyTorch to use the MPS device if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def benchmark(model_name: str) -> pd.DataFrame:
    """
    Benchmarks a model on the instance recognition dataset.

    Args:
        model_name (str): Name of the model to benchmark.

    Returns:
        pd.DataFrame: DataFrame containing the benchmark results.
    """
    results = []

    print(f"Benchmarking {model_name} on instance-recognition...")

    # Load data
    data_path = "/Users/woodj/Desktop/fishy-business/data/REIMS.xlsx"
    if not os.path.exists(data_path):
        data_path = "/vol/ecrg-solar/woodj4/fishy-business/data/REIMS.xlsx"

    data_config = DataConfig(
        batch_size=32,
        data_path=data_path,
    )
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data(data_config)
    filtered_data = preprocessor.filter_data(data, data_config.dataset_name)
    features = filtered_data.drop("m/z", axis=1).to_numpy()
    labels = preprocessor.encode_labels(filtered_data, data_config.dataset_name)

    dataset = SiameseDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=data_config.batch_size)
    
    n_features = features.shape[1]

    # --- Training Time Measurement ---
    config = ContrastiveConfig(encoder_type=model_name, input_dim=n_features)
    model = create_backbone_encoder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    start_time = time.time()
    model.train()
    for x1, x2, _ in data_loader:
        x1 = x1.to(device)
        optimizer.zero_grad()
        output = model(x1)
        loss = criterion(output, torch.zeros_like(output))
        loss.backward()
        optimizer.step()
        break # Only need one batch for timing
    training_time = time.time() - start_time

    # --- Inference Time Measurement ---
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for x1, x2, _ in data_loader:
            x1 = x1.to(device)
            model(x1)
            break # Only need one batch for timing
    inference_time = time.time() - start_time

    # Memory usage
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())

    # Model parameters
    num_params = sum(p.numel() for p in model.parameters())

    results.append(
        {
            "model": model_name,
            "dataset": "instance-recognition",
            "training_time": training_time,
            "inference_time": inference_time,
            "model_size_mb": model_size / 1e6,
            "num_params": num_params,
        }
    )

    return pd.DataFrame(results)


if __name__ == "__main__":
    """Main entry point for benchmarking models."""
    parser = argparse.ArgumentParser(description="Benchmark models.")
    parser.add_argument("models", type=str, nargs="+", help="The models to benchmark.")
    args = parser.parse_args()

    all_results = []
    for model_name in args.models:
        results_df = benchmark(model_name)
        all_results.append(results_df)
        print(results_df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("benchmark_results_contrastive.csv", index=False)
    print("All benchmark results saved to benchmark_results_contrastive.csv")

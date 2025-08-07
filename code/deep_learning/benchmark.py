import argparse
import time
import torch
import numpy as np
import pandas as pd
import os
from classifiers.clf.data import load_dataset
from deep_learning.train import train_model, transfer_learning
from models import cnn, dense, lstm, rcnn, tcn, transformer, wavenet, MOE, ensemble
from deep_learning import pre_training
from torch.optim import AdamW

# Configure PyTorch to use the MPS device if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_model_instance(model_name, n_features, n_classes, device) -> torch.nn.Module:
    """Returns an instance of the specified model.

    Args:
        model_name (str): Name of the model to instantiate.
        n_features (int): Number of input features.
        n_classes (int): Number of output classes.
        device (torch.device): Device to run the model on.

    Returns:
        nn.Module: An instance of the specified model.
    """
    if model_name == "cnn":
        return cnn.CNN(input_size=n_features, num_classes=n_classes)
    elif model_name == "dense":
        return dense.Dense(input_dim=n_features, output_dim=n_classes)
    elif model_name == "lstm":
        return lstm.LSTM(input_size=n_features, output_size=n_classes)
    elif model_name == "rcnn":
        return rcnn.RCNN(input_size=n_features, num_classes=n_classes)
    elif model_name == "tcn":
        return tcn.TCN(input_dim=n_features, output_dim=n_classes)
    elif model_name == "transformer":
        return transformer.Transformer(
            input_dim=n_features, output_dim=n_classes, num_heads=1, hidden_dim=128
        )
    elif model_name == "transformer_pretrained":
        model = transformer.Transformer(
            input_dim=n_features, output_dim=n_classes, num_heads=1, hidden_dim=128
        )
        checkpoint_path = "transformer_checkpoint_msm.pth"
        if not os.path.exists(checkpoint_path):
            print("Pre-trained model not found. Pre-training now...")
            # Pre-training logic directly in benchmark.py
            original_fc = model.fc_out
            model.fc_out = torch.nn.Linear(model.fc_out.in_features, n_features).to(
                device
            )
            optimizer = AdamW(model.parameters(), lr=1e-4)
            dummy_X = torch.rand(10, n_features).to(device)
            dummy_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(dummy_X, dummy_X), batch_size=5
            )
            config = pre_training.PreTrainingConfig(
                n_features=n_features,
                device=device,
                num_epochs=1,
                file_path=checkpoint_path,
            )
            pre_trainer = pre_training.PreTrainer(model, config, optimizer)
            pre_trainer.pre_train_masked_spectra(dummy_loader)
            torch.save(model.state_dict(), checkpoint_path)
            model.fc_out = original_fc

        # Load the pre-trained weights, but ignore the final layer
        pretrained_dict = torch.load(checkpoint_path)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and not k.startswith("fc_out")
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model
    elif model_name == "moe":
        return MOE(
            input_dim=n_features, output_dim=n_classes, num_heads=1, hidden_dim=128
        )
    elif model_name == "ensemble":
        return ensemble.Ensemble(
            input_dim=n_features, output_dim=n_classes, hidden_dim=128
        )
    elif model_name == "wavenet":
        return wavenet.WaveNet(input_dim=n_features, output_dim=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def benchmark(model_name) -> pd.DataFrame:
    """
    Benchmarks a model on the four classification tasks.

    Args:
        model_name (str): Name of the model to benchmark.
        warmup_epochs (int): Number of warm-up epochs.

    Returns:
        pd.DataFrame: DataFrame containing the benchmark results.
    """
    datasets = ["species", "part", "oil", "cross-species"]
    results = []

    for dataset_name in datasets:
        print(f"Benchmarking {model_name} on {dataset_name}...")

        # Load data
        X, y = load_dataset(dataset_name)
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X).float(), torch.from_numpy(y).long()
            ),
            batch_size=32,
        )

        # --- Warm-up ---
        if warmup_epochs > 0:
            print(f"Running {warmup_epochs} warm-up epochs...")
            # Warm-up on a dummy model instance
            warmup_model = get_model_instance(
                model_name, n_features, n_classes, device
            ).to(device)
            train_model(
                warmup_model,
                train_loader,
                torch.nn.CrossEntropyLoss(),
                torch.optim.Adam(warmup_model.parameters()),
                num_epochs=warmup_epochs,
                n_splits=1,
                n_runs=1,
            )
            with torch.no_grad():
                warmup_model(torch.from_numpy(X).float().to(device))

        # --- Training Time Measurement ---
        # Create a fresh model for accurate training benchmark
        model = get_model_instance(model_name, n_features, n_classes, device).to(device)

        start_time = time.time()
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X).float(), torch.from_numpy(y).long()
            ),
            batch_size=32,
        )
        train_model(
            model,
            train_loader,
            torch.nn.CrossEntropyLoss(),
            torch.optim.Adam(model.parameters()),
            num_epochs=1,
            n_splits=1,
            n_runs=1,
        )
        training_time = time.time() - start_time

        # --- Inference Time Measurement ---
        start_time = time.time()
        with torch.no_grad():
            model(torch.from_numpy(X).float().to(device))
        inference_time = time.time() - start_time

        # Memory usage
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())

        # Model parameters
        num_params = sum(p.numel() for p in model.parameters())

        results.append(
            {
                "model": model_name,
                "dataset": dataset_name,
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
        results_df = benchmark(model_name, warmup_epochs=args.warmup_epochs)
        all_results.append(results_df)
        print(results_df)
        results_df.to_csv(f"benchmark_results_{model_name}.csv", index=False)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("benchmark_results_all_transformers.csv", index=False)
    print("All benchmark results saved to benchmark_results_all_transformers.csv")

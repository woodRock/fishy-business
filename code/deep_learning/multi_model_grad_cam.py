"""
This script performs a comparative 1D Grad-CAM analysis on multiple models
(Transformer, MOE, LSTM) trained on mass spectrometry data. It identifies the
top 10 most important features for each model and visualizes them on a single graph.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import List

# Import models and utilities
from models import Transformer, MOE, LSTM, CNN
from deep_learning.util import create_data_module
from deep_learning.grad_cam import GradCAM, prepare_data_loaders


def load_model(
    model_class: nn.Module,
    model_path: os.path,
    device: torch.device,
    load_if_exists: bool = True,
    **kwargs,
) -> nn.Module:
    """Loads a trained model from a file.

    Args:
        model_class: The class of the model to load.
        model_path: Path to the saved model file.
        device: Device to load the model onto (CPU or GPU).
        load_if_exists: Whether to load the model if it exists, or initialize a new one.
        **kwargs: Additional parameters for model initialization.

    Returns:
        An instance of the model class loaded with the saved state or a new instance.

    Raises:
        FileNotFoundError: If the model file does not exist and load_if_exists is True
        Exception: If there is an error loading the model state.
    """
    model = model_class(**kwargs)
    if os.path.exists(model_path) and load_if_exists:
        try:
            print(f"Loading pre-trained model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(
                f"Error loading model from {model_path}: {e}. Initializing a new model."
            )
            model = model_class(**kwargs)  # Re-initialize model if loading fails
    elif not load_if_exists:
        print(f"Pre-trained model not found at {model_path}. Initializing a new model.")
        model = model_class(**kwargs)  # Initialize new model if not loading from file
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.to(device)
    model.eval()
    return model


def get_top_features(cam_map: torch.tensor, top_k: int = 10) -> np.ndarray:
    """Extracts the top k features from a CAM map.

    Args:
        cam_map: The CAM map tensor (1D for 1D Grad-CAM).
        top_k: Number of top features to extract.

    Returns:
        Indices of the top k features in the CAM map.
    Raises:
        ValueError: If top_k is greater than the number of features in cam_map.
    """
    # Get the indices of the top k features
    top_k_indices = torch.topk(cam_map, top_k).indices.cpu().numpy()
    return top_k_indices


def plot_top_features_comparison(
    feature_sets: List[List[int]],
    model_names: List[str],
    mass_to_charge_ratios: np.ndarray,
    output_path: os.path,
) -> None:
    """Plots the top features for multiple models on the same graph.

    Args:
        feature_sets: List of arrays containing top features for each model.
        model_names: List of model names corresponding to the feature sets.
        mass_to_charge_ratios: Array of mass-to-charge ratios.
        output_path: Path to save the comparison plot.
    """
    plt.figure(figsize=(15, 8))

    jitter_strength = 0.1

    for i, (features, name) in enumerate(zip(feature_sets, model_names)):
        y_values = np.random.normal(loc=i, scale=jitter_strength, size=len(features))
        # Ensure features are integers for indexing
        features = np.array(features).astype(int)
        plt.scatter(
            mass_to_charge_ratios[features], y_values, label=name, alpha=0.7, s=50
        )

    plt.yticks(range(len(model_names)), model_names)
    plt.xlabel("Feature Index (m/z)")
    plt.title("Top 10 Most Important Features per Model (Grad-CAM)")
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved top features comparison plot to {output_path}")

def save_top_features_to_table(
    feature_sets: List[List[int]], model_names: List[str], dataframe: pd.DataFrame
) -> None:
    """Saves the top features to a CSV file.

    Args:
        feature_sets: List of arrays containing top features for each model.
        model_names: List of model names corresponding to the feature sets.
        dataframe: DataFrame containing the original data for reference.
    """
    if dataframe is not None:
        # Get the m/z values from the dataframe columns
        mz_values = dataframe.columns[1:] 

        top_features_data = []
        for i, features in enumerate(feature_sets):
            # Map feature indices to m/z values
            top_mz_values = [mz_values[j] for j in features]
            top_features_data.append(top_mz_values)

        top_features_df = pd.DataFrame(
            {
                "Model": model_names,
                "Top Features (m/z)": top_features_data,
            }
        )
        top_features_df.to_csv("gradcam_results/top_features.csv", index=False)
        print("Saved top features to gradcam_results/top_features.csv")
    else:
        print("No dataframe available to save top features.")

def main():
    """Main function to run the multi-model Grad-CAM analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    data_module = create_data_module(
        file_path="/Users/woodj/Desktop/fishy-business/data/REIMS.xlsx",
        dataset_name="cross-species",
        batch_size=32,
    )
    data_module.setup()
    data_loader = data_module.get_train_dataloader()
    dataframe = data_module.get_train_dataframe()
    _, _, test_loader = prepare_data_loaders(data_loader, val_split=0.2, test_split=0.1)

    sample_features, _ = next(iter(test_loader))
    feature_dim = sample_features.shape[1]
    output_dim = 3  # For 'part' dataset

    # --- Model Configurations ---
    model_configs = {
        "Transformer": {
            "class": Transformer,
            "path": "model/trained_transformer.pt",
            "params": {
                "input_dim": feature_dim,
                "output_dim": output_dim,
                "num_heads": 4,
                "hidden_dim": 128,
                "num_layers": 7,
                "dropout": 0.1,
            },
            "target_layer_getter": lambda m: m.attention_layers[-1],
        },
        "MOE": {
            "class": MOE,
            "path": "model/trained_moe.pt",  # Assuming this path exists
            "params": {
                "input_dim": feature_dim,
                "output_dim": output_dim,
                "num_heads": 4,
                "hidden_dim": 128,
                "num_layers": 7,
                "num_experts": 4,
                "k": 2,
            },
            "target_layer_getter": lambda m: m.attention_layers[-1],
        },
        "LSTM": {
            "class": LSTM,
            "path": "model/trained_lstm.pt",  # Assuming this path exists
            "params": {
                "input_size": feature_dim,
                "output_size": output_dim,
                "hidden_size": 128,
                "num_layers": 2,
            },
            "target_layer_getter": lambda m: m.lstm,
        },
        "CNN": {
            "class": CNN,
            "path": "model/trained_cnn.pt",  # Assuming this path exists
            "params": {"input_size": feature_dim, "num_classes": output_dim},
            "target_layer_getter": lambda m: m.conv_layers[8],
        },
    }

    all_top_features = []
    model_names = []

    # --- Analysis Loop ---
    for name, config in model_configs.items():
        print(f"--- Analyzing {name} ---")
        try:
            model = load_model(
                config["class"],
                config["path"],
                device,
                load_if_exists=config.get("load_if_exists", True),
                **config["params"],
            )

            # If the model was newly initialized (i.e., not loaded), train it
            if not os.path.exists(config["path"]) or not config.get(
                "load_if_exists", True
            ):
                print(f"Training {name} model...")
                # Assuming train_model is available in the scope or imported
                from deep_learning.grad_cam import train_model

                model = train_model(
                    model=model,
                    train_loader=data_module.get_train_dataloader(),
                    val_loader=prepare_data_loaders(data_module.get_train_dataloader())[
                        1
                    ],  # Get validation loader
                    device=device,
                    num_epochs=100,  # Train for a few epochs for demonstration
                    save_path=config["path"],
                )
            target_layer = config["target_layer_getter"](model)

            grad_cam = GradCAM(model, target_layer)

            # Use a single batch for analysis
            features, _ = next(iter(test_loader))
            features = features.to(device)

            cam_maps = grad_cam.generate_cam(features)

            # Average CAM map across the batch
            avg_cam_map = cam_maps.mean(dim=0)

            top_features = get_top_features(avg_cam_map, top_k=10)
            all_top_features.append(top_features)
            model_names.append(name)

            grad_cam.remove_hooks()

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred during {name} analysis: {e}")

    mass_to_charge_ratios = dataframe.columns[1:].tolist()  # Adjust as needed
    # Convert mass-to-charge ratios to float if they are strings
    mass_to_charge_ratios = np.array(mass_to_charge_ratios)

    # --- Plotting ---
    os.makedirs("gradcam_results", exist_ok=True)
    if all_top_features:
        plot_top_features_comparison(
            all_top_features, 
            model_names, 
            mass_to_charge_ratios, # Assuming first column is m/z
            "gradcam_results/top_features_comparison.png"
        )

    # --- Save to table ---
    if all_top_features:
        save_top_features_to_table(all_top_features, model_names, dataframe)

if __name__ == "__main__":
    main()

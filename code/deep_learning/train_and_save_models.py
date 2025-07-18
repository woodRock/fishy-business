
"""
This script trains and saves the Transformer, MOE, and LSTM models.
"""

import torch
import torch.nn as nn
import os

from models import Transformer, MOE, LSTM, CNN
from .util import create_data_module
from .train import train_model

def main():
    """ Main function to train and save models.
    
    This function initializes the data module, sets up the training environment,
    and trains the Transformer, MOE, and LSTM models. It saves the trained models
    to specified paths."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_module = create_data_module(
        file_path="/Users/woodj/Desktop/fishy-business/data/REIMS.xlsx",
        dataset_name="cross-species",
        batch_size=32,
    )
    data_module.setup()
    train_loader = data_module.get_train_dataloader()

    sample_features, sample_labels = next(iter(train_loader))
    feature_dim = sample_features.shape[1]
    output_dim = sample_labels.shape[1]

    os.makedirs("model", exist_ok=True)

    model_configs = {
        "Transformer": {
            "class": Transformer,
            "path": "model/trained_transformer.pt",
            "params": {"input_dim": feature_dim, "output_dim": output_dim, "num_heads": 4, "hidden_dim": 128, "num_layers": 7, "dropout": 0.1},
        },
        "MOE": {
            "class": MOE,
            "path": "model/trained_moe.pt",
            "params": {"input_dim": feature_dim, "output_dim": output_dim, "num_heads": 4, "hidden_dim": 128, "num_layers": 7, "num_experts": 4, "k": 2},
        },
        "LSTM": {
            "class": LSTM,
            "path": "model/trained_lstm.pt",
            "params": {"input_size": feature_dim, "output_size": output_dim, "hidden_size": 128, "num_layers": 2},
        },
        "CNN": {
            "class": CNN,
            "path": "model/trained_cnn.pt",  # Assuming this path exists
            "params": {"input_size": feature_dim, "num_classes": output_dim},
        }
    }

    for name, config in model_configs.items():
        print(f"--- Training {name} ---")
        model = config["class"](**config["params"])
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        trained_model, _ = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=100,  # Using 1 epoch for demonstration
            patience=5,
            n_splits=5, # Using 2 splits for demonstration
            n_runs=1, # Using 1 run for demonstration
            device=device,
        )

        torch.save(trained_model.state_dict(), config["path"])
        print(f"Saved trained {name} model to {config['path']}")

if __name__ == "__main__":
    main()

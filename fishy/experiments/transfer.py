# -*- coding: utf-8 -*-
"""
Transfer learning module for deep learning models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)
import seaborn as sns
import copy
import os
from typing import List, Dict
from pathlib import Path

from fishy.engine.training_loops import train_with_tracking
from fishy.data.module import create_data_module
from fishy._core.factory import create_model
from fishy._core.config import TrainingConfig

def run_sequential_transfer_learning(
    model_name: str,
    transfer_datasets: List[str],
    target_dataset: str,
    num_epochs_transfer: int = 10,
    num_epochs_finetune: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    finetune_lr: float = 5e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_intermediate: bool = False,
    val_split: float = 0.2,
    file_path: str = None
):
    """
    Performs sequential transfer learning across multiple datasets.
    """
    history = {"transfer": {}, "finetune": {}}
    device_obj = torch.device(device)
    data_path = file_path if file_path else str(Path(__file__).resolve().parent.parent.parent / "data" / "REIMS.xlsx")

    # Initial data module to get dimensions
    data_module = create_data_module(
        file_path=data_path,
        dataset_name=transfer_datasets[0],
        batch_size=batch_size,
    )
    data_module.setup()
    input_dim = data_module.get_input_dim()
    
    # We need to handle the output_dim changing.
    # Initial model creation
    config = TrainingConfig(
        file_path="", model=model_name, dataset=transfer_datasets[0], 
        run=0, output="", data_augmentation=False, 
        masked_spectra_modelling=False, next_spectra_prediction=False,
        next_peak_prediction=False, spectrum_denoising_autoencoding=False,
        peak_parameter_regression=False, spectrum_segment_reordering=False,
        contrastive_transformation_invariance_learning=False,
        early_stopping=0, dropout=0.2, label_smoothing=0.1,
        epochs=num_epochs_transfer, learning_rate=learning_rate, batch_size=batch_size,
        hidden_dimension=128, num_layers=4, num_heads=4,
        num_augmentations=0, noise_level=0.0, shift_enabled=False,
        scale_enabled=False, k_folds=1
    )

    # Determine initial num_classes
    from fishy.experiments.deep_training import ModelTrainer
    num_classes = ModelTrainer.N_CLASSES_PER_DATASET.get(transfer_datasets[0], 2)
    model = create_model(config, input_dim, num_classes).to(device_obj)
    print(f"Model {model_name} initialized on {device}")

    # Sequential transfer learning
    for i, dataset_name in enumerate(transfer_datasets):
        print(f"\nPhase {i+1}: Transfer Learning on '{dataset_name}'")
        
        data_module = create_data_module(
            file_path=data_path,
            dataset_name=dataset_name,
            batch_size=batch_size,
        )
        data_module.setup()
        dataset = data_module.get_dataset()
        
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Adapt output layer if necessary
        # This assumes models have a 'fc_out' or similar. 
        # For simplicity in this wrapper, we try to adapt based on common names.
        current_num_classes = ModelTrainer.N_CLASSES_PER_DATASET.get(dataset_name, 2)
        
        output_layer = None
        for attr in ['fc_out', 'classifier', 'fc']:
            if hasattr(model, attr):
                output_layer = getattr(model, attr)
                layer_name = attr
                break
        
        if output_layer and isinstance(output_layer, nn.Linear):
            if output_layer.out_features != current_num_classes:
                in_features = output_layer.in_features
                new_layer = nn.Linear(in_features, current_num_classes).to(device_obj)
                setattr(model, layer_name, new_layer)
                print(f"Adapted {layer_name} to {current_num_classes} classes")

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

        dataset_history = train_with_tracking(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs_transfer,
            device=device_obj,
        )
        history["transfer"][dataset_name] = dataset_history

        if save_intermediate:
            torch.save(model.state_dict(), f"model_transfer_{dataset_name}.pt")

    # Fine-tuning
    print(f"\nFinal Phase: Fine-tuning on '{target_dataset}'")
    data_module = create_data_module(
        file_path=data_path,
        dataset_name=target_dataset,
        batch_size=batch_size,
    )
    data_module.setup()
    target_data = data_module.get_dataset()
    
    val_size = int(val_split * len(target_data))
    train_size = len(target_data) - val_size
    train_dataset, val_dataset = random_split(target_data, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    current_num_classes = ModelTrainer.N_CLASSES_PER_DATASET.get(target_dataset, 2)
    # Adapt output layer again
    output_layer = None
    for attr in ['fc_out', 'classifier', 'fc']:
        if hasattr(model, attr):
            output_layer = getattr(model, attr)
            layer_name = attr
            break
    
    if output_layer and isinstance(output_layer, nn.Linear):
        if output_layer.out_features != current_num_classes:
            in_features = output_layer.in_features
            new_layer = nn.Linear(in_features, current_num_classes).to(device_obj)
            setattr(model, layer_name, new_layer)
            print(f"Adapted {layer_name} to {current_num_classes} classes")

    optimizer = AdamW(model.parameters(), lr=finetune_lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    finetune_history = train_with_tracking(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs_finetune,
        device=device_obj,
    )
    history["finetune"][target_dataset] = finetune_history

    # Final Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device_obj), y.to(device_obj)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            true_labels = torch.argmax(y, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    final_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f"\nFinal Balanced Accuracy: {final_acc*100:.2f}%")
    
    return model, history

def visualize_transfer_results(history: Dict):
    # Reuse the logic from original transfer_learning.py for plotting if needed
    pass
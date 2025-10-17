#!/bin/bash

# This script runs a comparative experiment of different ordinal classification
# methods across various encoder architectures on the 'oil' dataset.

# --- Configuration ---
MODELS=("transformer" "cnn" "ensemble" "kan" "mamba" "moe" "rcnn" "vae")
DATASET="oil"
KFOLDS=5
EPOCHS=50
BASE_OUTPUT_NAME="logs/ordinal_comparison"

# --- Experiment Loop ---

echo "Starting Ordinal Classification Experiment..."

for model in "${MODELS[@]}"; do
    echo "----------------------------------------------------"
    echo "Running experiments for MODEL: $model"
    echo "----------------------------------------------------"

    # --- Method 1: Regression ---
    echo "
[1/3] Running REGRESSION-BASED method..."
    python3 -m deep_learning.main \
        --model "$model" \
        --dataset "$DATASET" \
        --k-folds "$KFOLDS" \
        --epochs "$EPOCHS" \
        --output "${BASE_OUTPUT_NAME}_${model}_regression" \
        --regression

    # --- Method 2: CORAL ---
    echo "
[2/3] Running CORAL method..."
    python3 -m deep_learning.main \
        --model "$model" \
        --dataset "$DATASET" \
        --k-folds "$KFOLDS" \
        --epochs "$EPOCHS" \
        --output "${BASE_OUTPUT_NAME}_${model}_coral" \
        --use-coral

    # --- Method 3: Cumulative Link ---
    echo "
[3/3] Running CUMULATIVE LINK method..."
    python3 -m deep_learning.main \
        --model "$model" \
        --dataset "$DATASET" \
        --k-folds "$KFOLDS" \
        --epochs "$EPOCHS" \
        --output "${BASE_OUTPUT_NAME}_${model}_cumulative_link" \
        --use-cumulative-link

done

echo "
All experiments completed."

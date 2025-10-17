#!/bin/bash

# This script re-runs the experiments that failed during the previous execution.

# --- Configuration ---
MODELS=("transformer" "cnn" "ensemble" "kan" "mamba" "moe" "rcnn" "vae")
DATASET="oil"
KFOLDS=5
EPOCHS=50
BASE_OUTPUT_NAME="logs/ordinal_comparison"

echo "Starting Re-run of Failed Ordinal Classification Experiments..."

# --- Run all models with cumulative_link ---
echo "
Running all models with CUMULATIVE LINK method..."
for model in "${MODELS[@]}"; do
    echo "
Running CUMULATIVE LINK for MODEL: $model"
    python3 -m deep_learning.main \
        --model "$model" \
        --dataset "$DATASET" \
        --k-folds "$KFOLDS" \
        --epochs "$EPOCHS" \
        --output "${BASE_OUTPUT_NAME}_${model}_cumulative_link" \
        --use-cumulative-link
done

# --- Run vae with regression and coral ---
echo "
Running VAE model with REGRESSION and CORAL methods..."

echo "
[VAE] Running REGRESSION-BASED method..."
python3 -m deep_learning.main \
    --model "vae" \
    --dataset "$DATASET" \
    --k-folds "$KFOLDS" \
    --epochs "$EPOCHS" \
    --output "${BASE_OUTPUT_NAME}_vae_regression" \
    --regression

echo "
[VAE] Running CORAL method..."
python3 -m deep_learning.main \
    --model "vae" \
    --dataset "$DATASET" \
    --k-folds "$KFOLDS" \
    --epochs "$EPOCHS" \
    --output "${BASE_OUTPUT_NAME}_vae_coral" \
    --use-coral

# --- Run mamba with regression and coral ---
echo "
Running MAMBA model with REGRESSION and CORAL methods..."

echo "
[MAMBA] Running REGRESSION-BASED method..."
python3 -m deep_learning.main \
    --model "mamba" \
    --dataset "$DATASET" \
    --k-folds "$KFOLDS" \
    --epochs "$EPOCHS" \
    --output "${BASE_OUTPUT_NAME}_mamba_regression" \
    --regression

echo "
[MAMBA] Running CORAL method..."
python3 -m deep_learning.main \
    --model "mamba" \
    --dataset "$DATASET" \
    --k-folds "$KFOLDS" \
    --epochs "$EPOCHS" \
    --output "${BASE_OUTPUT_NAME}_mamba_coral" \
    --use-coral

echo "
All failed experiments have been re-run."

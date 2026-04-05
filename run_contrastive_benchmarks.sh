#!/bin/bash

# Contrastive and Binary Classification Experiment Runner for Fishy Business
# Queues all models 30 times with fixed seeds using task spooler (task -G)

# Configuration - Methods, Epochs, Dataset, and Seeds are all configurable here
SEEDS=(
    42 123 777 999 2024 1337 555 101 888 1234
    111 222 333 444 666 808 909 1010 1111 2025
    7 13 21 34 55 89 144 233 377 610
)

DATASET=${1:-batch-detection} # Default to batch-detection
EPOCHS=${2:-30}               # Default to 30 epochs
DIRECTORY="/vol/ecrg-solar/woodj4/fishy-business"

# 1. Traditional Methods (Binary Classification)
TRADITIONAL_MODELS=("knn" "dt" "lr" "lda" "nb" "rf" "svm" "xgb" "opls-da")

# 2. Deep Learning Methods (Binary Classification)
DEEP_MODELS=("transformer" "cnn" "ensemble" "moe" "rcnn" "lstm" "kan")

# 3. Contrastive Learning Methods (SimCLR objective)
CONTRASTIVE_MODELS=("simclr")
ENCODERS=("transformer" "cnn" "ensemble" "moe" "rcnn" "lstm" "kan")

echo "Queuing experiment sweep: ${#SEEDS[@]} seeds x (Traditional + Deep + Contrastive)"
echo "Dataset: $DATASET | Epochs: $EPOCHS"
echo "Usage: ./run_contrastive_benchmarks.sh [dataset_name] [epochs]"

# --- Traditional Models ---
for model in "${TRADITIONAL_MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        task -G 1 -d "$DIRECTORY" fishy train -m "$model" -d "$DATASET" --seed "$seed" --benchmark --figures --wandb-log
    done
done

# --- Deep Learning Models (Binary) ---
for model in "${DEEP_MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        task -G 1 -d "$DIRECTORY" fishy train -m "$model" -d "$DATASET" --seed "$seed" --epochs "$EPOCHS" --benchmark --figures --wandb-log
    done
done

# --- Contrastive Learning Models (SimCLR) ---
for model in "${CONTRASTIVE_MODELS[@]}"; do
    for encoder in "${ENCODERS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            task -G 1 -d "$DIRECTORY" fishy train -m "$model" -d "$DATASET" --encoder "$encoder" --seed "$seed" --epochs "$EPOCHS" --benchmark --figures --wandb-log
        done
    done
done

echo "Success: All tasks have been added to the task spooler queue."
echo "Use 'task' to view the queue status."

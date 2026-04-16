#!/bin/bash
# Benchmark: Muon SOTA (Muon + EMA + Warmup)
# Purpose: Isolate Muon performance for statistical comparison.

DATASETS=("species" "parts" "oil" "cross-species")
RUNS=30
WANDB_PROJECT="transformer-recipe-test"

echo "Starting isolated Muon SOTA Benchmark Suite..."
echo "Total Runs: 120 (4 datasets * 1 config * 30 runs)"

for DATASET in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Processing Dataset: ${DATASET}"
    echo "========================================"

    # The Muon SOTA Recipe (The Ultimate Hypothesis)
    # Muon + Warmup + EMA
    echo "Running Muon SOTA (Muon + EMA + Warmup)..."
    task -G 1 fishy train -m augformer \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e 1000 \
        --optimizer muon \
        --lr 0.001 \
        --patience 1000 \
        --warmup-epochs 5 \
        --ema \
        --ema-decay 0.999 \
        --normalize \
        --wandb-log \
        --benchmark

done

echo "Muon isolated Benchmark Suite Complete. Results in WandB project: ${WANDB_PROJECT}"

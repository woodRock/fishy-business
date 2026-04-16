#!/bin/bash
# Benchmark: Standard AugFormer vs. Long-Schedule EMA AugFormer
# Purpose: Test if EMA + long training improves robustness/performance.

DATASETS=("species" "parts" "oil" "xsa")
RUNS=30
WANDB_PROJECT="ema-hypothesis-test"

echo "Starting EMA Hypothesis Benchmark Suite..."
echo "Total Runs: 240 (4 datasets * 2 configs * 30 runs)"

for DATASET in "${DATASETS[@]}"; do
    echo "Processing Dataset: ${DATASET}"

    # 1. Standard AugFormer (Baseline)
    # 100 epochs max, 20 patience early stopping, no EMA
    echo "Running Standard Baseline for ${DATASET}..."
    task -G 1 fishy train -m augformer \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e 100 \
        --patience 20 \
        --normalize \
        --wandb-log \
        --benchmark

    # 2. EMA Hypothesis Configuration
    # 1000 epochs, no early stopping (patience 1000), EMA enabled
    echo "Running EMA Hypothesis Configuration for ${DATASET}..."
    task -G 1 fishy train -m augformer \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e 1000 \
        --patience 1000 \
        --ema \
        --ema-decay 0.999 \
        --normalize \
        --wandb-log \
        --benchmark

done

echo "Benchmark Suite Complete. Results logged to WandB project: ${WANDB_PROJECT}"

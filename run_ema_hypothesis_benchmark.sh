#!/bin/bash
# Benchmark: Standard vs. Long EMA vs. Full SOTA Recipe
# Purpose: Test synergistic interactions of Warmup, EMA, and TTT.

DATASETS=("species" "part" "oil" "cross-species")
RUNS=30
WANDB_PROJECT="fishy-business"

echo "Starting Full SOTA Recipe Benchmark Suite..."
echo "Total Runs: 360 (4 datasets * 3 configs * 30 runs)"

for DATASET in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Processing Dataset: ${DATASET}"
    echo "========================================"

    # 1. Standard AugFormer (Baseline)
    echo "Run 1/3: Standard Baseline..."
    task -G 1 fishy train -m augformer \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e 100 \
        --patience 20 \
        --normalize \
        --wandb-log \
        --benchmark

    # 2. Long EMA (The Stability Hypothesis)
    echo "Run 2/3: Long EMA + Warmup..."
    task -G 1 fishy train -m augformer \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e 1000 \
        --patience 1000 \
        --warmup-epochs 5 \
        --ema \
        --ema-decay 0.999 \
        --normalize \
        --wandb-log \
        --benchmark

   done

echo "Benchmark Suite Complete. Results in WandB project: ${WANDB_PROJECT}"

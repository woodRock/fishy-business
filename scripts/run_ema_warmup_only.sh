#!/bin/bash
# Benchmark: AugFormer EMA + Warmup (AdamW)
# Purpose: Focused run for the Stability-only hypothesis.

DATASETS=("species" "part" "oil" "cross-species")
RUNS=30

echo "Starting AugFormer EMA + Warmup (AdamW) Runs..."
echo "Total Runs: 120 (4 datasets * 30 runs)"

for DATASET in "${DATASETS[@]}"; do
    echo "Running EMA + Warmup for ${DATASET}..."
    task -G 1 -n "augformer_ema_warmup_${DATASET}" fishy train -m augformer \
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

echo "Runs complete. Results logged to WandB."

#!/bin/bash
# Benchmark: DeepSeekV4 SOTA (12 Layers)
# Purpose: Test if 12-layer DeepSeekV4 architecture provides superior reasoning and stability.

DATASETS=("species" "part" "oil" "cross-species")
RUNS=30

echo "Starting DeepSeekV4 (12-layer) Benchmark Suite..."
echo "Total Runs: 120 (4 datasets * 30 runs)"

for DATASET in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Processing Dataset: ${DATASET}"
    echo "========================================"

    # DeepSeekV4 Configuration
    # Uses 12 layers by default, along with MLA, MHC, and Engram.
    task -G 1 -n "deepseekv4_12l_${DATASET}" fishy train -m deepseekv4 \
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

echo "DeepSeekV4 Benchmark Runs Complete. Ready for statistical analysis."

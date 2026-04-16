#!/bin/bash
# Benchmark: Frankenstein SOTA
# Purpose: Test synergistic effect of MLA, MHC, and Engram on top of Muon/EMA.

DATASETS=("species" "part" "oil" "cross-species")
RUNS=30

echo "Starting Frankenstein Benchmark Suite..."
echo "Total Runs: 120 (4 datasets * 30 runs)"

for DATASET in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Processing Dataset: ${DATASET}"
    echo "========================================"

    # The Frankenstein Configuration
    # Muon + EMA + Warmup + MLA + MHC + Engram
    task -G 1 -n "augformer_frankenstein_${DATASET}" fishy train -m augformer \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e 1000 \
        --optimizer muon \
        --lr 0.001 \
        --patience 1000 \
        --warmup-epochs 5 \
        --ema \
        --ema-decay 0.999 \
        --mla \
        --mhc \
        --engram \
        --normalize \
        --wandb-log \
        --benchmark

done

echo "Frankenstein Runs Complete. Ready for statistical analysis."

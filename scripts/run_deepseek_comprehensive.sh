#!/bin/bash
# Benchmark: DeepSeekV4 Comprehensive Study
# Purpose: Evaluate DeepSeekV4 (with Engram Rejection Gate) across all datasets with 3 incremental optimizations.

DATASETS=("species" "part" "oil" "cross-species")
RUNS=30
EPOCHS=1000
PATIENCE=1000

echo "Starting DeepSeekV4 Comprehensive Benchmark Suite..."
echo "Total Runs: 360 (4 datasets * 3 configurations * 30 runs)"

for DATASET in "${DATASETS[@]}"; do
    echo "============================================================"
    echo " DATASET: ${DATASET}"
    echo "============================================================"

    # 1. Baseline DeepSeekV4 (AdamW, no EMA, no Warmup)
    echo "Running Configuration 1: Baseline (AdamW)..."
    task -G 1 -n "ds_baseline_${DATASET}" fishy train -m deepseekv4 \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e "${EPOCHS}" \
        --patience "${PATIENCE}" \
        --normalize \
        --wandb-log \
        --benchmark

    # 2. DeepSeekV4 + Warmup + EMA (AdamW)
    echo "Running Configuration 2: Warmup + EMA (AdamW)..."
    task -G 1 -n "ds_warmup_ema_${DATASET}" fishy train -m deepseekv4 \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e "${EPOCHS}" \
        --patience "${PATIENCE}" \
        --warmup-epochs 5 \
        --ema \
        --ema-decay 0.999 \
        --normalize \
        --wandb-log \
        --benchmark

    # 3. DeepSeekV4 + Warmup + EMA + Muon (Full SOTA)
    echo "Running Configuration 3: Warmup + EMA + Muon (SOTA)..."
    task -G 1 -n "ds_muon_sota_${DATASET}" fishy train -m deepseekv4 \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e "${EPOCHS}" \
        --optimizer muon \
        --lr 0.001 \
        --patience "${PATIENCE}" \
        --warmup-epochs 5 \
        --ema \
        --ema-decay 0.999 \
        --normalize \
        --wandb-log \
        --benchmark

done

echo "Comprehensive DeepSeekV4 Benchmarks Complete."

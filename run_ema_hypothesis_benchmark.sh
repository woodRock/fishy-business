#!/bin/bash
# Full Experimental Sweep: AugFormer vs. GatedMLP vs. SOTA Recipes
# Purpose: Comprehensive benchmark of architectural baselines vs. synergistic regularization.

DATASETS=("species" "part" "oil" "cross-species")
RUNS=30

echo "Starting Full Experimental Sweep..."
echo "Total Runs: 480 (4 datasets * 4 configs * 30 runs)"

for DATASET in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Processing Dataset: ${DATASET}"
    echo "========================================"

    # 1. Baseline AugFormer
    echo "Run 1/4: AugFormer Baseline (AdamW, 100e)..."
    task -G 1 -n "augformer_baseline_${DATASET}" fishy train -m augformer \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e 100 \
        --patience 20 \
        --normalize \
        --wandb-log \
        --benchmark

    # 2. Warmup + EMA AugFormer
    echo "Run 2/4: AugFormer EMA + Warmup (AdamW, 1000e)..."
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

    # 3. Baseline GatedMLP
    echo "Run 3/4: GatedMLP Baseline (AdamW, 100e)..."
    task -G 1 -n "gatedmlp_baseline_${DATASET}" fishy train -m gatedmlp \
        -d "${DATASET}" \
        -N "${RUNS}" \
        -e 100 \
        --patience 20 \
        --normalize \
        --wandb-log \
        --benchmark

    # 4. Warmup + EMA + Muon AugFormer (SOTA Recipe)
    echo "Run 4/4: AugFormer Muon SOTA (Muon + EMA + Warmup)..."
    task -G 1 -n "augformer_muon_sota_${DATASET}" fishy train -m augformer \
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

echo "Experimental Sweep Complete. All results logged to WandB."

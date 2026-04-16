#!/bin/bash
# Full Experimental Sweep: AugFormer vs. GatedMLP vs. SOTA Recipes
# Purpose: Balanced ablation study to isolate Optimizer vs. Regularization effects.

DATASETS=("species" "part" "oil" "cross-species")
RUNS=30

echo "Starting Balanced Experimental Sweep..."
echo "Total Runs: 720 (4 datasets * 6 configs * 30 runs)"

for DATASET in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Processing Dataset: ${DATASET}"
    echo "========================================"

    # --- AugFormer Suite ---
    echo "Run 1/6: AugFormer Baseline (AdamW, 100e)..."
    task -G 1 -n "aug_base_${DATASET}" fishy train -m augformer -d "${DATASET}" -N "${RUNS}" -e 100 --patience 20 --normalize --wandb-log --benchmark

    echo "Run 2/6: AugFormer EMA + Warmup (AdamW, 1000e)..."
    task -G 1 -n "aug_ema_${DATASET}" fishy train -m augformer -d "${DATASET}" -N "${RUNS}" -e 1000 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark

    echo "Run 3/6: AugFormer Muon SOTA (Muon + EMA + Warmup)..."
    task -G 1 -n "aug_muon_${DATASET}" fishy train -m augformer -d "${DATASET}" -N "${RUNS}" -e 1000 --optimizer muon --lr 0.001 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark

    # --- GatedMLP Suite ---
    echo "Run 4/6: GatedMLP Baseline (AdamW, 100e)..."
    task -G 1 -n "gate_base_${DATASET}" fishy train -m gatedmlp -d "${DATASET}" -N "${RUNS}" -e 100 --patience 20 --normalize --wandb-log --benchmark

    echo "Run 5/6: GatedMLP EMA + Warmup (AdamW, 1000e)..."
    task -G 1 -n "gate_ema_${DATASET}" fishy train -m gatedmlp -d "${DATASET}" -N "${RUNS}" -e 1000 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark

    echo "Run 6/6: GatedMLP Muon SOTA (Muon + EMA + Warmup)..."
    task -G 1 -n "gate_muon_${DATASET}" fishy train -m gatedmlp -d "${DATASET}" -N "${RUNS}" -e 1000 --optimizer muon --lr 0.001 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark

done

echo "Experimental Sweep Complete. All results logged to WandB."

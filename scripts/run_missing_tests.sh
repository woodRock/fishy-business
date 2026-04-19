#!/bin/bash
# Script to fill in missing runs for EMA and Muon configurations to reach N=30.

# 1. Species Dataset
echo "Completing Species..."
task -G 1 -n "augformer_ema_missing_species" fishy train -m augformer -d species -N 14 -e 1000 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark
task -G 1 -n "augformer_muon_missing_species" fishy train -m augformer -d species -N 21 -e 1000 --optimizer muon --lr 0.001 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark

# 2. Part Dataset
echo "Completing Part..."
task -G 1 -n "augformer_ema_missing_part" fishy train -m augformer -d part -N 10 -e 1000 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark
task -G 1 -n "augformer_muon_missing_part" fishy train -m augformer -d part -N 16 -e 1000 --optimizer muon --lr 0.001 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark

# 3. Oil Dataset
echo "Completing Oil..."
task -G 1 -n "augformer_ema_missing_oil" fishy train -m augformer -d oil -N 15 -e 1000 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark
task -G 1 -n "augformer_muon_missing_oil" fishy train -m augformer -d oil -N 21 -e 1000 --optimizer muon --lr 0.001 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark

# 4. Cross-Species Dataset
echo "Completing Cross-Species..."
task -G 1 -n "augformer_ema_missing_cross-species" fishy train -m augformer -d cross-species -N 22 -e 1000 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark
task -G 1 -n "augformer_muon_missing_cross-species" fishy train -m augformer -d cross-species -N 22 -e 1000 --optimizer muon --lr 0.001 --patience 1000 --warmup-epochs 5 --ema --ema-decay 0.999 --normalize --wandb-log --benchmark

echo "All missing runs queued."

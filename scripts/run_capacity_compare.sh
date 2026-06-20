#!/usr/bin/env bash
set -euo pipefail

# Reproduce final 100-epoch model-capacity experiments.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_capacity_compare.sh

cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/results

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/capacity_small_e100.yaml \
  --output_dir outputs/results/capacity_small_e100 \
  2>&1 | tee outputs/logs/capacity_small_e100_run.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/vit_baseline_e100.yaml \
  --output_dir outputs/results/vit_baseline_e100 \
  2>&1 | tee outputs/logs/vit_baseline_e100_run.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/capacity_large_e100.yaml \
  --output_dir outputs/results/capacity_large_e100 \
  2>&1 | tee outputs/logs/capacity_large_e100_run.log

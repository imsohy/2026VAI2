#!/usr/bin/env bash
set -euo pipefail

# Reproduce final 100-epoch CNN and ViT baseline experiments.
# Usage examples:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_baselines.sh
#   bash scripts/run_baselines.sh

cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/results

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/vit_baseline_e100.yaml \
  --output_dir outputs/results/vit_baseline_e100 \
  2>&1 | tee outputs/logs/vit_baseline_e100_run.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/cnn_baseline_e100.yaml \
  --output_dir outputs/results/cnn_baseline_e100 \
  2>&1 | tee outputs/logs/cnn_baseline_e100_run.log

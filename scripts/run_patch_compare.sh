#!/usr/bin/env bash
set -euo pipefail

# Reproduce final 100-epoch patch-size experiments.
# p=4 is represented by vit_baseline_e100, so patch_p4_e100 is not run separately.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_patch_compare.sh

cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/results

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/patch_p2_e100.yaml \
  --output_dir outputs/results/patch_p2_e100 \
  2>&1 | tee outputs/logs/patch_p2_e100_run.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/vit_baseline_e100.yaml \
  --output_dir outputs/results/vit_baseline_e100 \
  2>&1 | tee outputs/logs/vit_baseline_e100_run.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/patch_p8_e100.yaml \
  --output_dir outputs/results/patch_p8_e100 \
  2>&1 | tee outputs/logs/patch_p8_e100_run.log

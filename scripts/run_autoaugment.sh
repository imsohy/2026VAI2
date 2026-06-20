#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/results

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/twist_autoaugment.yaml \
  --output_dir outputs/results/twist_autoaugment \
  2>&1 | tee outputs/logs/twist_autoaugment_run.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/twist_autoaugment_e100.yaml \
  --output_dir outputs/results/twist_autoaugment_e100 \
  2>&1 | tee outputs/logs/twist_autoaugment_e100_run.log

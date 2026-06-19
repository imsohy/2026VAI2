#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python src/train.py --config configs/capacity_small.yaml --output_dir outputs/results/capacity_small 2>&1 | tee outputs/logs/capacity_small.log
python src/train.py --config configs/vit_baseline.yaml --output_dir outputs/results/capacity_base 2>&1 | tee outputs/logs/capacity_base.log
python src/train.py --config configs/capacity_large.yaml --output_dir outputs/results/capacity_large 2>&1 | tee outputs/logs/capacity_large.log

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python src/train.py --config configs/vit_baseline.yaml --output_dir outputs/results/vit_baseline 2>&1 | tee outputs/logs/vit_baseline.log
python src/train.py --config configs/cnn_baseline.yaml --output_dir outputs/results/cnn_baseline 2>&1 | tee outputs/logs/cnn_baseline.log

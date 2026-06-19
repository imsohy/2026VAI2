#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python src/train.py --config configs/vit_baseline.yaml --output_dir outputs/results/twist_cls_reference
python src/train.py --config configs/twist_meanpool.yaml --output_dir outputs/results/twist_meanpool

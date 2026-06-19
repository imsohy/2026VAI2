#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python src/train.py --config configs/patch_p2.yaml --output_dir outputs/results/patch_p2 2>&1 | tee outputs/logs/patch_p2.log
python src/train.py --config configs/patch_p4.yaml --output_dir outputs/results/patch_p4 2>&1 | tee outputs/logs/patch_p4.log
python src/train.py --config configs/patch_p8.yaml --output_dir outputs/results/patch_p8 2>&1 | tee outputs/logs/patch_p8.log

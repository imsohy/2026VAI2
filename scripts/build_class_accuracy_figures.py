#!/usr/bin/env python3
"""Build class-wise test accuracy figures for all final report models.

This script is intended for the report's class-wise error/accuracy analysis.
It collects per-class test accuracy for the seven final experiments and writes
both CSV tables and report-ready plots.

It can work in two modes:
1. If outputs/results/<experiment>/class_accuracy.csv already exists, it reads it.
2. If the CSV is missing but checkpoint_best.ckpt exists, it re-evaluates the
   saved checkpoint on CIFAR-10 test set and creates the CSV automatically.

Recommended usage from project root:

    python scripts/build_class_accuracy_figures.py

Outputs:
    outputs/results/<experiment>/class_accuracy.csv
    outputs/results/<experiment>/figures/class_accuracy.png
    report_figures/class_accuracy_all_models.csv
    report_figures/class_accuracy_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CIFAR10_CLASSES, get_cifar10_loaders  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from lit_module import LitImageClassifier  # noqa: E402
from utils import load_config  # noqa: E402


# Seven final report experiments. The patch-size-4 result is vit_baseline_e100.
EXPERIMENTS: Dict[str, Dict[str, str]] = {
    "cnn_baseline_e100": {"label": "CNN baseline"},
    "vit_baseline_e100": {"label": "ViT baseline / p4"},
    "patch_p2_e100": {"label": "Patch size 2"},
    "patch_p8_e100": {"label": "Patch size 8"},
    "capacity_small_e100": {"label": "Small ViT"},
    "capacity_large_e100": {"label": "Large ViT"},
    "twist_autoaugment_e100": {"label": "ViT + AutoAugment"},
}


FINAL_ORDER = list(EXPERIMENTS.keys())


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] Requested {requested}, but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def compute_class_accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> pd.DataFrame:
    """Return class-wise correct/total/accuracy table for CIFAR-10."""
    true_list = list(y_true)
    pred_list = list(y_pred)
    rows: List[dict] = []
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        total = sum(1 for y in true_list if y == class_idx)
        correct = sum(1 for y, p in zip(true_list, pred_list) if y == class_idx and p == class_idx)
        accuracy = correct / total if total > 0 else 0.0
        rows.append(
            {
                "class_index": class_idx,
                "class_name": class_name,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
            }
        )
    return pd.DataFrame(rows)


def plot_single_model_class_accuracy(df: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.sort_values("class_index").copy()
    values = plot_df["accuracy"].astype(float) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(plot_df["class_name"], values)
    ax.set_title(title)
    ax.set_xlabel("CIFAR-10 class")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def plot_class_accuracy_heatmap(all_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_order = [EXPERIMENTS[name]["label"] for name in FINAL_ORDER]
    pivot = all_df.pivot(index="label", columns="class_name", values="accuracy")
    pivot = pivot.reindex(index=label_order, columns=CIFAR10_CLASSES)
    matrix = pivot.astype(float).to_numpy() * 100.0

    fig, ax = plt.subplots(figsize=(13, 6.5))
    im = ax.imshow(matrix, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Test accuracy (%)")

    ax.set_title("Class-wise Test Accuracy across Final Experiments")
    ax.set_xlabel("CIFAR-10 class")
    ax.set_ylabel("Experiment")
    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(label_order)))
    ax.set_yticklabels(label_order)

    # Annotate each cell so the report can be read even after resizing.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def load_checkpoint_path(exp_dir: Path) -> Optional[Path]:
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            ckpt = metrics.get("best_checkpoint_path")
            if ckpt:
                ckpt_path = Path(ckpt)
                if ckpt_path.exists():
                    return ckpt_path
                # If metrics stored an old absolute path, fall back to the local output dir.
                fallback = exp_dir / ckpt_path.name
                if fallback.exists():
                    return fallback
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to parse {metrics_path}: {exc}")

    fallback = exp_dir / "checkpoint_best.ckpt"
    if fallback.exists():
        return fallback
    return None


def evaluate_experiment(exp_name: str, exp_dir: Path, device: torch.device) -> Optional[pd.DataFrame]:
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        print(f"[WARN] Missing config: {config_path}")
        return None

    ckpt_path = load_checkpoint_path(exp_dir)
    if ckpt_path is None:
        print(f"[WARN] Missing checkpoint for {exp_name}. Expected {exp_dir / 'checkpoint_best.ckpt'}")
        return None

    print(f"[INFO] Evaluating {exp_name} from {ckpt_path}")
    config = load_config(config_path)
    _, _, test_loader = get_cifar10_loaders(config)
    lit_model = LitImageClassifier.load_from_checkpoint(str(ckpt_path), config=config)
    model = lit_model.model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_model(model, test_loader, criterion, device, collect_examples=False)
    return compute_class_accuracy(test_metrics["y_true"], test_metrics["y_pred"])


def load_or_build_class_accuracy(
    exp_name: str,
    results_dir: Path,
    device: torch.device,
    force_evaluate: bool = False,
) -> Optional[pd.DataFrame]:
    exp_dir = results_dir / exp_name
    csv_path = exp_dir / "class_accuracy.csv"
    fig_path = exp_dir / "figures" / "class_accuracy.png"

    if csv_path.exists() and not force_evaluate:
        df = pd.read_csv(csv_path)
        print(f"[OK] Loaded {csv_path}")
    else:
        df = evaluate_experiment(exp_name, exp_dir, device)
        if df is None:
            return None
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved {csv_path}")

    # Always regenerate the per-experiment figure so style stays consistent.
    title = f"Class-wise Test Accuracy: {EXPERIMENTS[exp_name]['label']}"
    plot_single_model_class_accuracy(df, fig_path, title)

    df = df.copy()
    df.insert(0, "experiment", exp_name)
    df.insert(1, "label", EXPERIMENTS[exp_name]["label"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build class-wise CIFAR-10 accuracy plots for final experiments.")
    parser.add_argument("--results_dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--output_dir", type=Path, default=Path("report_figures"))
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--force_evaluate",
        action="store_true",
        help="Re-evaluate checkpoints even if class_accuracy.csv already exists.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[INFO] Using device: {device}")

    all_rows: List[pd.DataFrame] = []
    for exp_name in FINAL_ORDER:
        df = load_or_build_class_accuracy(
            exp_name=exp_name,
            results_dir=args.results_dir,
            device=device,
            force_evaluate=args.force_evaluate,
        )
        if df is not None:
            all_rows.append(df)

    if not all_rows:
        raise SystemExit("No class-wise accuracy data was generated. Check checkpoints and output paths.")

    all_df = pd.concat(all_rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_csv = args.output_dir / "class_accuracy_all_models.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"[OK] Saved {all_csv}")

    plot_class_accuracy_heatmap(all_df, args.output_dir / "class_accuracy_heatmap.png")

    print("\nDone. Use report_figures/class_accuracy_heatmap.png in the Results/Error Analysis section.")


if __name__ == "__main__":
    main()

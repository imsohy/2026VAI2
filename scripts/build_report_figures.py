#!/usr/bin/env python3
"""
Build integrated report figures from saved experiment outputs.

This script does not train models. It only reads files already produced by
src/train.py, especially:
  - outputs/results/<experiment>/metrics.json
  - outputs/results/<experiment>/train_log.csv

It writes report-ready aggregate figures to:
  - report_figures/

Recommended usage:
  cd /media/cine/First/HWPJ2/2026VAI2
  python scripts/build_report_figures.py

The script is intentionally independent from checkpoints. Large files such as
.ckpt/.pt are not required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# Final report experiments. vit_baseline_e100 is also the patch-size-4 result.
EXPERIMENTS: Dict[str, Dict[str, object]] = {
    "cnn_baseline_e100": {
        "label": "CNN baseline",
        "family": "cnn_vs_vit",
    },
    "vit_baseline_e100": {
        "label": "ViT baseline / p4",
        "family": "baseline",
        "patch_size": 4,
        "token_count": 64,
        "capacity": "base",
    },
    "twist_autoaugment_e100": {
        "label": "ViT + AutoAugment",
        "family": "twist",
        "patch_size": 4,
        "token_count": 64,
        "capacity": "base",
    },
    "twist_meanpool_e100": {
        "label": "ViT + mean pooling",
        "family": "twist",
        "patch_size": 4,
        "token_count": 64,
        "capacity": "base",
    },
    "patch_p2_e100": {
        "label": "Patch size 2",
        "family": "patch",
        "patch_size": 2,
        "token_count": 256,
    },
    "patch_p8_e100": {
        "label": "Patch size 8",
        "family": "patch",
        "patch_size": 8,
        "token_count": 16,
    },
    "capacity_small_e100": {
        "label": "Small ViT",
        "family": "capacity",
        "capacity": "small",
    },
    "capacity_large_e100": {
        "label": "Large ViT",
        "family": "capacity",
        "capacity": "large",
    },
}

PATCH_EXPERIMENTS = ["patch_p2_e100", "vit_baseline_e100", "patch_p8_e100"]
CAPACITY_EXPERIMENTS = ["capacity_small_e100", "vit_baseline_e100", "capacity_large_e100"]
TWIST_EXPERIMENTS = ["vit_baseline_e100", "twist_meanpool_e100", "twist_autoaugment_e100"]
CNN_VIT_EXPERIMENTS = ["cnn_baseline_e100", "vit_baseline_e100", "twist_autoaugment_e100"]


def load_metrics(results_dir: Path, exp_name: str) -> Optional[dict]:
    path = results_dir / exp_name / "metrics.json"
    if not path.exists():
        print(f"[WARN] Missing metrics: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_log(results_dir: Path, exp_name: str) -> Optional[pd.DataFrame]:
    path = results_dir / exp_name / "train_log.csv"
    if not path.exists():
        print(f"[WARN] Missing train log: {path}")
        return None
    df = pd.read_csv(path)
    expected_cols = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    if list(df.columns) != expected_cols and len(df.columns) >= 5:
        df = df.iloc[:, :5]
        df.columns = expected_cols
    for col in expected_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def collect_summary(results_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for exp_name, meta in EXPERIMENTS.items():
        metrics = load_metrics(results_dir, exp_name)
        log = load_log(results_dir, exp_name)
        if metrics is None:
            continue

        final_train_acc = None
        final_val_acc = None
        final_train_loss = None
        final_val_loss = None
        if log is not None and not log.empty:
            final_row = log.dropna(subset=["epoch"]).iloc[-1]
            final_train_acc = final_row.get("train_acc")
            final_val_acc = final_row.get("val_acc")
            final_train_loss = final_row.get("train_loss")
            final_val_loss = final_row.get("val_loss")

        rows.append(
            {
                "experiment": exp_name,
                "label": meta.get("label", exp_name),
                "family": meta.get("family", ""),
                "patch_size": meta.get("patch_size"),
                "token_count": meta.get("token_count"),
                "capacity": meta.get("capacity"),
                "test_acc": metrics.get("test_acc"),
                "test_loss": metrics.get("test_loss"),
                "best_val_acc": metrics.get("best_val_acc"),
                "num_parameters": metrics.get("num_parameters"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                "training_time_readable": metrics.get("training_time_readable"),
                "final_train_acc": final_train_acc,
                "final_val_acc": final_val_acc,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "final_train_val_gap": (
                    final_train_acc - final_val_acc
                    if final_train_acc is not None and final_val_acc is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def save_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    xlabel: str = "",
    y_as_percent: bool = False,
    annotate: bool = True,
) -> None:
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        print(f"[WARN] No data for {output_path.name}")
        return

    values = plot_df[y_col].astype(float)
    if y_as_percent:
        values = values * 100.0

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(plot_df)), 5))
    bars = ax.bar(plot_df[x_col].astype(str), values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    if annotate:
        for bar, value in zip(bars, values):
            if pd.isna(value):
                continue
            label = f"{value:.2f}" if y_as_percent else f"{value:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def save_line_plot(
    results_dir: Path,
    exp_names: Iterable[str],
    title: str,
    output_path: Path,
    metric: str = "val_acc",
    y_as_percent: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for exp_name in exp_names:
        log = load_log(results_dir, exp_name)
        if log is None or log.empty or metric not in log.columns:
            continue
        series = log[["epoch", metric]].dropna()
        if series.empty:
            continue
        y = series[metric].astype(float)
        if y_as_percent:
            y = y * 100.0
        label = str(EXPERIMENTS.get(exp_name, {}).get("label", exp_name))
        ax.plot(series["epoch"], y, marker="o", markersize=2, linewidth=1.5, label=label)
        plotted = True

    if not plotted:
        print(f"[WARN] No curve data for {output_path.name}")
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ") + (" (%)" if y_as_percent else ""))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def save_overfit_curve(results_dir: Path, exp_name: str, output_path: Path) -> None:
    log = load_log(results_dir, exp_name)
    if log is None or log.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["train_acc", "val_acc"]:
        series = log[["epoch", metric]].dropna()
        if not series.empty:
            ax.plot(
                series["epoch"],
                series[metric].astype(float) * 100.0,
                marker="o",
                markersize=2,
                linewidth=1.5,
                label=metric,
            )
    ax.set_title(f"Train vs Validation Accuracy: {EXPERIMENTS[exp_name]['label']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def save_param_scatter(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.dropna(subset=["num_parameters", "test_acc"]).copy()
    if plot_df.empty:
        print(f"[WARN] No data for {output_path.name}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = plot_df["num_parameters"].astype(float) / 1_000_000
    y = plot_df["test_acc"].astype(float) * 100.0
    ax.scatter(x, y, s=80)
    for _, row in plot_df.iterrows():
        ax.annotate(
            str(row["label"]),
            (row["num_parameters"] / 1_000_000, row["test_acc"] * 100.0),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_title("Parameter Count vs Test Accuracy")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def save_time_scatter(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.dropna(subset=["training_time_seconds", "test_acc"]).copy()
    if plot_df.empty:
        print(f"[WARN] No data for {output_path.name}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = plot_df["training_time_seconds"].astype(float) / 60.0
    y = plot_df["test_acc"].astype(float) * 100.0
    ax.scatter(x, y, s=80)
    for _, row in plot_df.iterrows():
        ax.annotate(
            str(row["label"]),
            (row["training_time_seconds"] / 60.0, row["test_acc"] * 100.0),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_title("Training Time vs Test Accuracy")
    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def subset(df: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    order = {name: idx for idx, name in enumerate(names)}
    out = df[df["experiment"].isin(order.keys())].copy()
    out["_order"] = out["experiment"].map(order)
    return out.sort_values("_order").drop(columns=["_order"])


def write_inventory(fig_dir: Path) -> None:
    inventory = """# Report Figure Inventory

Generated by `scripts/build_report_figures.py`.

## Integrated figures

- `summary_metrics.csv`: final metric table for report tables.
- `model_test_accuracy.png`: test accuracy of all final experiments.
- `cnn_vs_vit_accuracy.png`: CNN baseline vs ViT baseline vs AutoAugment ViT.
- `training_time_comparison.png`: training time comparison for all final experiments.
- `patch_accuracy.png`: patch size vs test accuracy.
- `patch_training_time.png`: patch size vs training time.
- `patch_token_count.png`: patch size vs token count.
- `param_vs_accuracy.png`: parameter count vs test accuracy.
- `time_vs_accuracy.png`: training time vs test accuracy.
- `train_val_gap.png`: final train-validation accuracy gap.
- `vit_baseline_overfit_curve.png`: baseline ViT train/validation curve.
- `autoaugment_overfit_curve.png`: AutoAugment ViT train/validation curve.
- `cnn_vit_val_curve.png`: validation curve comparison between CNN, ViT, and AutoAugment ViT.
- `patch_val_curve.png`: validation curve comparison across patch sizes.
- `capacity_val_curve.png`: validation curve comparison across model capacities.
- `twist_accuracy.png`: baseline, mean pooling, and AutoAugment comparison.
- `twist_val_curve.png`: validation curve comparison across twist experiments.

## Per-experiment figures to place manually if needed

Each experiment directory under `outputs/results/<experiment>/figures/` may contain:

- `learning_curve.png`
- `learning_curve_loss.png`
- `confusion_matrix.png`
- `wrong_examples.png`
- `patch_visualization.png` or dataset sample images if generated
"""
    path = fig_dir / "figure_inventory.md"
    path.write_text(inventory, encoding="utf-8")
    print(f"[OK] Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build integrated report figures from experiment outputs.")
    parser.add_argument("--results_dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--output_dir", type=Path, default=Path("report_figures"))
    args = parser.parse_args()

    results_dir = args.results_dir
    fig_dir = args.output_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_summary(results_dir)
    if summary.empty:
        raise SystemExit(f"No experiment metrics found under {results_dir}")

    summary_csv = fig_dir / "summary_metrics.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[OK] Saved {summary_csv}")

    # Overall comparison figures.
    save_bar(
        summary,
        "label",
        "test_acc",
        "Final Test Accuracy by Experiment",
        "Test Accuracy (%)",
        fig_dir / "model_test_accuracy.png",
        y_as_percent=True,
    )
    save_bar(
        summary,
        "label",
        "training_time_seconds",
        "Training Time by Experiment",
        "Training Time (seconds)",
        fig_dir / "training_time_comparison.png",
    )
    save_bar(
        summary,
        "label",
        "final_train_val_gap",
        "Final Train-Validation Accuracy Gap",
        "Train Acc - Val Acc",
        fig_dir / "train_val_gap.png",
    )
    save_param_scatter(summary, fig_dir / "param_vs_accuracy.png")
    save_time_scatter(summary, fig_dir / "time_vs_accuracy.png")

    # CNN vs ViT.
    cnn_vit = subset(summary, CNN_VIT_EXPERIMENTS)
    save_bar(
        cnn_vit,
        "label",
        "test_acc",
        "CNN vs ViT vs AutoAugment ViT",
        "Test Accuracy (%)",
        fig_dir / "cnn_vs_vit_accuracy.png",
        y_as_percent=True,
    )
    save_line_plot(results_dir, CNN_VIT_EXPERIMENTS, "Validation Accuracy: CNN vs ViT", fig_dir / "cnn_vit_val_curve.png")

    # Patch-size ablation.
    patch_df = subset(summary, PATCH_EXPERIMENTS)
    patch_df["patch_label"] = patch_df["patch_size"].apply(lambda p: f"p={int(p)}" if pd.notna(p) else "p=?")
    save_bar(
        patch_df,
        "patch_label",
        "test_acc",
        "Patch Size vs Test Accuracy",
        "Test Accuracy (%)",
        fig_dir / "patch_accuracy.png",
        xlabel="Patch size",
        y_as_percent=True,
    )
    save_bar(
        patch_df,
        "patch_label",
        "training_time_seconds",
        "Patch Size vs Training Time",
        "Training Time (seconds)",
        fig_dir / "patch_training_time.png",
        xlabel="Patch size",
    )
    save_bar(
        patch_df,
        "patch_label",
        "token_count",
        "Patch Size vs Token Count",
        "Number of Patch Tokens",
        fig_dir / "patch_token_count.png",
        xlabel="Patch size",
    )
    save_line_plot(results_dir, PATCH_EXPERIMENTS, "Validation Accuracy by Patch Size", fig_dir / "patch_val_curve.png")

    # Capacity ablation.
    capacity_df = subset(summary, CAPACITY_EXPERIMENTS)
    save_bar(
        capacity_df,
        "label",
        "test_acc",
        "Model Capacity vs Test Accuracy",
        "Test Accuracy (%)",
        fig_dir / "capacity_accuracy.png",
        y_as_percent=True,
    )
    save_line_plot(results_dir, CAPACITY_EXPERIMENTS, "Validation Accuracy by Model Capacity", fig_dir / "capacity_val_curve.png")

    # Twist comparison.
    twist_df = subset(summary, TWIST_EXPERIMENTS)
    save_bar(
        twist_df,
        "label",
        "test_acc",
        "Student Twist: Test Accuracy Comparison",
        "Test Accuracy (%)",
        fig_dir / "twist_accuracy.png",
        y_as_percent=True,
    )
    save_line_plot(results_dir, TWIST_EXPERIMENTS, "Validation Accuracy by Twist", fig_dir / "twist_val_curve.png")
    save_bar(
        subset(summary, ["vit_baseline_e100", "twist_autoaugment_e100"]),
        "label",
        "test_loss",
        "AutoAugment Effect: Test Loss",
        "Test Loss",
        fig_dir / "autoaugment_test_loss.png",
        y_as_percent=False,
    )
    save_bar(
        subset(summary, ["vit_baseline_e100", "twist_autoaugment_e100"]),
        "label",
        "test_acc",
        "AutoAugment Effect: Test Accuracy",
        "Test Accuracy (%)",
        fig_dir / "autoaugment_effect.png",
        y_as_percent=True,
    )

    # Explicit overfitting curves.
    save_overfit_curve(results_dir, "vit_baseline_e100", fig_dir / "vit_baseline_overfit_curve.png")
    save_overfit_curve(results_dir, "twist_autoaugment_e100", fig_dir / "autoaugment_overfit_curve.png")

    write_inventory(fig_dir)

    print("\nDone. Put report-ready aggregate figures from report_figures/ into the report.")


if __name__ == "__main__":
    main()

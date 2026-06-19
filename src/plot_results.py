"""Plotting utilities for report-ready figures."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from dataset import CIFAR10_CLASSES, denormalize


def plot_learning_curve(log_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(log_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Validation Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path.with_name(output_path.stem + "_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"], label="train accuracy")
    plt.plot(df["epoch"], df["val_acc"], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Validation Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CIFAR10_CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_CLASSES)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_wrong_examples(images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, output_path: str | Path, max_images: int = 16) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = denormalize(images[:max_images]).cpu()
    labels = labels[:max_images].cpu().tolist()
    preds = preds[:max_images].cpu().tolist()

    if images.numel() == 0:
        return

    n = min(len(images), max_images)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n:
            img = images[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].set_title(f"T: {CIFAR10_CLASSES[labels[i]]}\nP: {CIFAR10_CLASSES[preds[i]]}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_accuracy_bar(names: List[str], accuracies: List[float], output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(names, accuracies)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

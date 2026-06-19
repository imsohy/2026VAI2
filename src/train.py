"""Training entry point.

Run from project root:
python src/train.py --config configs/vit_baseline.yaml --output_dir outputs/results/vit_baseline
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from dataset import get_cifar10_loaders, save_dataset_samples, save_patch_visualization
from evaluate import evaluate_model
from models import build_model
from plot_results import plot_confusion_matrix, plot_learning_curve, save_wrong_examples
from utils import (
    copy_config,
    count_parameters,
    ensure_dir,
    format_seconds,
    get_device,
    get_gpu_name,
    load_config,
    save_json,
    set_seed,
    write_summary,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }


def build_optimizer(config, model):
    train_cfg = config["training"]
    name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("learning_rate", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 0.0)

    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    set_seed(config["training"].get("seed", 42), deterministic=config["training"].get("deterministic", True))
    copy_config(args.config, output_dir)

    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader = get_cifar10_loaders(config)

    # Report-ready visualizations generated once per experiment.
    if config.get("visualization", {}).get("save_dataset_samples", True):
        save_dataset_samples(val_loader, figures_dir / "dataset_samples.png")
    if config.get("visualization", {}).get("save_patch_visualization", True) and config["model"]["type"] == "vit":
        save_patch_visualization(
            val_loader,
            patch_size=config["model"].get("patch_size", 4),
            output_path=figures_dir / "patch_visualization.png",
        )

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)

    num_params = count_parameters(model)
    epochs = config["training"].get("epochs", 20)
    best_val_acc = -1.0
    best_epoch = -1
    best_ckpt_path = output_dir / "checkpoint_best.pt"
    log_path = output_dir / "train_log.csv"

    start_time = time.time()
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate_model(model, val_loader, criterion, device)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
            }
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} | "
                f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f}"
            )

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict(), "config": config}, best_ckpt_path)

    train_time = time.time() - start_time

    # Load best model before final test evaluation.
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, criterion, device, collect_examples=True)

    plot_learning_curve(log_path, figures_dir / "learning_curve.png")
    plot_confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], figures_dir / "confusion_matrix.png")
    save_wrong_examples(
        test_metrics["wrong_images"],
        test_metrics["wrong_labels"],
        test_metrics["wrong_preds"],
        figures_dir / "wrong_examples.png",
    )

    metrics = {
        "experiment_name": config.get("experiment_name", output_dir.name),
        "model_type": config["model"]["type"],
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "num_parameters": num_params,
        "training_time_seconds": train_time,
        "training_time_readable": format_seconds(train_time),
        "device": str(device),
        "gpu_name": get_gpu_name(),
        "seed": config["training"].get("seed", 42),
    }
    save_json(metrics, output_dir / "metrics.json")
    write_summary(output_dir / "summary.txt", metrics)
    print("\nFinal metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

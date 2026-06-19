"""Training entry point.

Run from project root:
python src/train.py --config configs/vit_baseline.yaml --output_dir outputs/results/vit_baseline

UVA Tutorial 15 adaptation notes
--------------------------------
This script keeps the UVA-style PyTorch Lightning training structure. Lightning
handles the epoch loop, batch loop, backward pass, optimizer step, validation
loop, and checkpoint callback. This project adds config-driven experiments and
report-ready csv/json/png outputs.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from dataset import get_cifar10_loaders, save_dataset_samples, save_patch_visualization
from evaluate import evaluate_model
from lit_module import LitImageClassifier
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


class EpochMetricRecorder(Callback):
    """Save one train/validation row per epoch as train_log.csv.

    Lightning already logs metrics internally. This callback converts the values
    required by the report into a flat csv with the same columns used by the
    plotting utilities.
    """

    def __init__(self, output_path: str | Path):
        super().__init__()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            writer.writeheader()

    @staticmethod
    def _to_float(value: Any):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    @staticmethod
    def _find_metric(metrics: Dict[str, Any], *names: str):
        for name in names:
            if name in metrics:
                return metrics[name]
        return None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = dict(trainer.callback_metrics)
        row = {
            "epoch": int(trainer.current_epoch) + 1,
            "train_loss": self._to_float(self._find_metric(metrics, "train_loss", "train_loss_epoch")),
            "train_acc": self._to_float(self._find_metric(metrics, "train_acc", "train_acc_epoch")),
            "val_loss": self._to_float(self._find_metric(metrics, "val_loss")),
            "val_acc": self._to_float(self._find_metric(metrics, "val_acc")),
        }
        with open(self.output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            writer.writerow(row)
            f.flush()


def resolve_trainer_device(config: Dict[str, Any]):
    requested = config["training"].get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return "gpu", 1
        return "cpu", 1
    if str(requested).startswith("cuda"):
        return "gpu", 1
    return "cpu", 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    set_seed(config["training"].get("seed", 42), deterministic=config["training"].get("deterministic", True))
    pl.seed_everything(config["training"].get("seed", 42), workers=True)
    copy_config(args.config, output_dir)

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

    lit_model = LitImageClassifier(config)
    num_params = count_parameters(lit_model.model)
    epochs = config["training"].get("epochs", 20)
    accelerator, devices = resolve_trainer_device(config)

    metric_recorder = EpochMetricRecorder(output_dir / "train_log.csv")
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="checkpoint_best",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    csv_logger = CSVLogger(save_dir=str(output_dir), name="lightning_csv")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        deterministic=config["training"].get("deterministic", True),
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback, lr_monitor, metric_recorder],
        enable_checkpointing=True,
        log_every_n_steps=10,
    )

    start_time = time.time()
    trainer.fit(lit_model, train_loader, val_loader)
    train_time = time.time() - start_time

    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path:
        best_ckpt_path = str(output_dir / "checkpoint_best.ckpt")
        trainer.save_checkpoint(best_ckpt_path)

    # Load best Lightning checkpoint, then run a report-oriented evaluation that
    # collects predictions, confusion matrix data, and wrong examples.
    best_lit_model = LitImageClassifier.load_from_checkpoint(best_ckpt_path, config=config)
    device = get_device(config["training"].get("device", "auto"))
    model = best_lit_model.model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_model(model, test_loader, criterion, device, collect_examples=True)

    plot_learning_curve(output_dir / "train_log.csv", figures_dir / "learning_curve.png")
    plot_confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], figures_dir / "confusion_matrix.png")
    save_wrong_examples(
        test_metrics["wrong_images"],
        test_metrics["wrong_labels"],
        test_metrics["wrong_preds"],
        figures_dir / "wrong_examples.png",
    )

    best_val_acc = checkpoint_callback.best_model_score
    best_val_acc_value = float(best_val_acc.detach().cpu()) if isinstance(best_val_acc, torch.Tensor) else None

    metrics = {
        "experiment_name": config.get("experiment_name", output_dir.name),
        "model_type": config["model"]["type"],
        "best_checkpoint_path": best_ckpt_path,
        "best_val_acc": best_val_acc_value,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "num_parameters": num_params,
        "training_time_seconds": train_time,
        "training_time_readable": format_seconds(train_time),
        "accelerator": accelerator,
        "devices": devices,
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

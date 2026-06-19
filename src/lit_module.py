"""PyTorch Lightning module for CIFAR-10 classification.

UVA Tutorial 15 adaptation notes
--------------------------------
The UVA Vision Transformer tutorial trains the CIFAR-10 ViT through a
LightningModule and a Trainer. This file keeps that high-level training style:
Lightning owns the epoch loop, batch loop, backward pass, optimizer step,
validation loop, and checkpoint callback execution.

This project-specific wrapper adds config-driven model construction so that
patch size, model capacity, and pooling strategy can be changed by YAML files.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn, optim
import pytorch_lightning as pl

from models import build_model


class LitImageClassifier(pl.LightningModule):
    """LightningModule wrapping the ViT or CNN model.

    The implementation intentionally keeps the training interface close to the
    UVA tutorial: `training_step`, `validation_step`, `test_step`, and
    `configure_optimizers` are handled by Lightning's Trainer.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config_dict = config
        self.model = build_model(config)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters({"config": config})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = self._accuracy(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = self._accuracy(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = self._accuracy(logits, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        train_cfg = self.config_dict["training"]
        name = train_cfg.get("optimizer", "adamw").lower()
        lr = train_cfg.get("learning_rate", 3e-4)
        weight_decay = train_cfg.get("weight_decay", 0.0)

        if name == "adamw":
            return optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        if name == "adam":
            return optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            return optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer: {name}")

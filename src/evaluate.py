"""Evaluation helpers."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_examples: bool = False,
    max_wrong_examples: int = 32,
) -> Dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    wrong_images = []
    wrong_labels = []
    wrong_preds = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

        if collect_examples and len(wrong_images) < max_wrong_examples:
            wrong_mask = preds != labels
            if wrong_mask.any():
                selected_images = images[wrong_mask].detach().cpu()
                selected_labels = labels[wrong_mask].detach().cpu()
                selected_preds = preds[wrong_mask].detach().cpu()
                remain = max_wrong_examples - len(wrong_images)
                for i in range(min(remain, selected_images.size(0))):
                    wrong_images.append(selected_images[i])
                    wrong_labels.append(selected_labels[i])
                    wrong_preds.append(selected_preds[i])

    result = {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if collect_examples:
        if wrong_images:
            result["wrong_images"] = torch.stack(wrong_images)
            result["wrong_labels"] = torch.stack(wrong_labels)
            result["wrong_preds"] = torch.stack(wrong_preds)
        else:
            result["wrong_images"] = torch.empty(0)
            result["wrong_labels"] = torch.empty(0, dtype=torch.long)
            result["wrong_preds"] = torch.empty(0, dtype=torch.long)
    return result

"""Dataset and visualization helpers.

UVA Tutorial 15 source connection:
- CIFAR-10 train/val/test split follows the 45k/5k/10k setup.
- Normalization constants and basic augmentation are inherited from the tutorial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models import img_to_patch

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_MEAN = [0.49139968, 0.48215841, 0.44653091]
CIFAR10_STD = [0.24703223, 0.24348513, 0.26158784]


def get_transforms(config: Dict):
    image_size = config["data"].get("image_size", 32)
    train_aug = config["data"].get("train_augmentation", True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = test_transform

    return train_transform, test_transform


def get_cifar10_loaders(config: Dict) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    data_dir = config["data"].get("data_dir", "data")
    batch_size = config["training"].get("batch_size", 128)
    num_workers = config["training"].get("num_workers", 4)
    seed = config["training"].get("seed", 42)
    pin_memory = torch.cuda.is_available()

    train_transform, test_transform = get_transforms(config)

    train_dataset = CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=data_dir, train=True, transform=test_transform, download=True)
    test_dataset = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)

    generator = torch.Generator().manual_seed(seed)
    train_set, _ = data.random_split(train_dataset, [45000, 5000], generator=generator)
    generator = torch.Generator().manual_seed(seed)
    _, val_set = data.random_split(val_dataset, [45000, 5000], generator=generator)

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=pin_memory, num_workers=num_workers,
    )
    val_loader = data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False,
        pin_memory=pin_memory, num_workers=num_workers,
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        pin_memory=pin_memory, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def save_dataset_samples(loader: data.DataLoader, output_path: str | Path, num_images: int = 16) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels = next(iter(loader))
    images = denormalize(images[:num_images])
    labels = labels[:num_images]

    grid = torchvision.utils.make_grid(images, nrow=4, padding=2)
    grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.title("CIFAR-10 dataset samples")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_patch_visualization(loader: data.DataLoader, patch_size: int, output_path: str | Path, num_images: int = 4) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images, _ = next(iter(loader))
    images = images[:num_images]
    raw_images = denormalize(images)
    patches = img_to_patch(raw_images, patch_size=patch_size, flatten_channels=False)

    fig, axes = plt.subplots(num_images, 1, figsize=(14, 3 * num_images))
    if num_images == 1:
        axes = [axes]
    fig.suptitle(f"Images as patch sequences, patch_size={patch_size}")

    num_patches = patches.shape[1]
    for i in range(num_images):
        grid = torchvision.utils.make_grid(patches[i], nrow=num_patches, normalize=True, pad_value=0.9)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(grid)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

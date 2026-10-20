"""
===============================================================================
File        : dataset.py
Project     : Handwritten Digit Recognition using CNN
Description :
    Load and preprocess EMNIST dataset with Strong Data Augmentation
    to bridge the domain gap for real-world photo inputs.
===============================================================================
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def transpose_image(t):
    return t.transpose(1, 2)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(transpose_image),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.85, 1.15)
    ),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Lambda(transpose_image),

    transforms.Normalize((0.1307,), (0.3081,))
])


def get_dataloader(batch_size=64):

    print("[INFO] Loading EMNIST Dataset...")

    train_dataset = datasets.EMNIST(
        root='./data',
        split='digits',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.EMNIST(
        root='./data',
        split='digits',
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"[INFO] Dataset loaded successfully! Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloader()
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
import os
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_dim: int, mean: list, std: list, train: bool = True):
    if train:
        return A.Compose([
            A.RandomResizedCrop(img_dim, img_dim, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_dim, img_dim),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class JewelleryDataset(Dataset):
    def __init__(self, root_dir: str, classes: list, transform=None):
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []

        for cls in classes:
            cls_dir = Path(root_dir) / cls
            if not cls_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                for img_path in cls_dir.glob(ext):
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import JewelleryDataset, get_transforms
from models import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name",   default="efficientnet_b3")
    p.add_argument("--img-dim",      type=int,   default=224)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-classes",  type=int,   default=42)
    p.add_argument("--classes",      type=str,   default="[]")
    p.add_argument("--val-split",    type=float, default=0.2)
    args, unknown = p.parse_known_args()
    print("Unknown SageMaker args:", unknown)
    return args


def main():
    args = parse_args()


    # SageMaker injects data/model paths via env vars
    train_dir  = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    classes = json.loads(args.classes)
    if not classes:
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print("Detected classes:", classes)
    model_dir  = os.environ.get("SM_MODEL_DIR",     "/opt/ml/model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    print("Train directory:", train_dir)
    print("Directory structure:")
    for root, dirs, files in os.walk(train_dir):
        print(f"{root} -> {len(files)} files")
    full_ds = JewelleryDataset(train_dir, classes, get_transforms(args.img_dim, imagenet_mean, imagenet_std, train=True))
    print("Dataset size:", len(full_ds))
    val_size   = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    val_ds.dataset.transform = get_transforms(args.img_dim, imagenet_mean, imagenet_std, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(args.model_name, args.num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss/len(train_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(model_dir) / "model.pth")

    # Save class index mapping
    with open(Path(model_dir) / "classes.json", "w") as f:
        json.dump(classes, f)

    print(f"Training done. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()

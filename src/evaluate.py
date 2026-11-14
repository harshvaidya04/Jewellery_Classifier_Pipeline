import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for SageMaker)
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Jewellery Classifier")

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory containing the saved model (model.pth + classes.json)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"),
        help="Root directory of the test/validation dataset (ImageFolder structure)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
        help="Directory to write evaluation outputs (metrics JSON, confusion matrix PNG)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-dim", type=int, default=300)
    parser.add_argument(
        "--model-name",
        type=str,
        default="efficientnet_b3",
        help="Architecture name (currently only efficientnet_b3 supported)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Report per-class metrics only for the K worst-performing classes",
    )
    return parser.parse_args()


def load_model(model_dir: str, num_classes: int, device: torch.device) -> nn.Module:
    """Load EfficientNet-B0 weights from model_dir/model.pth."""
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model_path = Path(model_dir) / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pth not found in {model_dir}")

    state = torch.load(model_path, map_location=device)
    # Support both raw state_dict and wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {model_path}")
    return model


def load_classes(model_dir: str):
    """Load class names from classes.json saved during training."""
    classes_path = Path(model_dir) / "classes.json"
    if classes_path.exists():
        with open(classes_path) as f:
            classes = json.load(f)
        print(f"[INFO] Loaded {len(classes)} classes from {classes_path}")
        return classes
    return None  # will fall back to dataset folder names


def build_dataloader(test_dir: str, img_dim: int, batch_size: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )
    print(f"[INFO] Test dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    return loader


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: torch.device):
    all_preds, all_labels, all_probs = [], [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * loader.batch_size} / {len(loader.dataset)} images")

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true, y_pred, class_names):
    metrics = {}

    # Overall metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
        }
    metrics["per_class"] = per_class

    return metrics


def print_summary(metrics, class_names, top_k=5):
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Accuracy          : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (macro) : {metrics['precision_macro']:.4f}")
    print(f"  Recall    (macro) : {metrics['recall_macro']:.4f}")
    print(f"  F1-score  (macro) : {metrics['f1_macro']:.4f}")
    print(f"  Precision (wtd)   : {metrics['precision_weighted']:.4f}")
    print(f"  Recall    (wtd)   : {metrics['recall_weighted']:.4f}")
    print(f"  F1-score  (wtd)   : {metrics['f1_weighted']:.4f}")
    print("=" * 60)

    # Worst performing classes by F1
    sorted_classes = sorted(
        metrics["per_class"].items(), key=lambda x: x[1]["f1"]
    )
    print(f"\n  Bottom {top_k} classes by F1-score:")
    print(f"  {'Class':<45} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*45} {'-'*10} {'-'*8} {'-'*8}")
    for cls, m in sorted_classes[:top_k]:
        print(f"  {cls:<45} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

    # Best performing classes by F1
    print(f"\n  Top {top_k} classes by F1-score:")
    print(f"  {'Class':<45} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*45} {'-'*10} {'-'*8} {'-'*8}")
    for cls, m in sorted_classes[-top_k:][::-1]:
        print(f"  {cls:<45} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
    print()


def save_confusion_matrix(y_true, y_pred, class_names, output_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)

    fig_size = max(20, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=(n <= 20),   # only show numbers for small matrices
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    out_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to {out_path}")

    # Also save normalised version
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_norm,
        annot=(n <= 20),
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (Normalised)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    out_path_norm = Path(output_dir) / "confusion_matrix_normalised.png"
    plt.savefig(out_path_norm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Normalised confusion matrix saved to {out_path_norm}")


def save_per_class_chart(metrics, output_dir: str):
    classes = list(metrics["per_class"].keys())
    f1_scores = [metrics["per_class"][c]["f1"] for c in classes]

    sorted_pairs = sorted(zip(classes, f1_scores), key=lambda x: x[1])
    classes_sorted, f1_sorted = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, max(8, len(classes) * 0.3)))
    colors = ["#d73027" if f < 0.5 else "#fee090" if f < 0.75 else "#1a9850" for f in f1_sorted]
    ax.barh(classes_sorted, f1_sorted, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="F1 = 0.5")
    ax.axvline(x=0.75, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="F1 = 0.75")
    ax.set_xlabel("F1-score", fontsize=11)
    ax.set_title("Per-Class F1-Score", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = Path(output_dir) / "per_class_f1.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Per-class F1 chart saved to {out_path}")


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load dataset first to get class names
    loader = build_dataloader(args.test_dir, args.img_dim, args.batch_size)
    class_names = load_classes(args.model_dir) or loader.dataset.classes
    num_classes = len(class_names)

    # Load model
    model = load_model(args.model_dir, num_classes, device)

    # Run inference
    print("\n[INFO] Running inference...")
    y_pred, y_true, y_probs = run_inference(model, loader, device)

    # Compute metrics
    print("\n[INFO] Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, class_names)

    # Print summary to stdout (captured by SageMaker logs)
    print_summary(metrics, class_names, top_k=args.top_k)

    # Full sklearn classification report
    print("\n[INFO] Full Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Save metrics JSON
    metrics_path = Path(args.output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {metrics_path}")

    # Save plots
    save_confusion_matrix(y_true, y_pred, class_names, args.output_dir)
    save_per_class_chart(metrics, args.output_dir)

    print("\n[INFO] Evaluation complete.")
    print(f"[INFO] Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
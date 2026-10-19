
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import accuracy_score, f1_score

from src.data.image_dataset import BloodCellDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    report_dir = root / "outputs" / "reports"
    model_dir = root / "outputs" / "models"
    manifest = report_dir / "splits.csv"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "class_to_idx.json") as f:
        class_to_idx = json.load(f)

    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    df = pd.read_csv(manifest)
    counts = df[df["split"] == "train"]["class_name"].value_counts()
    class_weights = torch.tensor(
        [counts.get(idx_to_class[i], 0) for i in range(num_classes)],
        dtype=torch.float32
    )
    if (class_weights == 0).any():
        raise ValueError("A class has no training samples.")
    class_weights = (
        class_weights.sum() / (num_classes * class_weights)
    ).to(device)

    train_ds = BloodCellDataset(split="train", csv_path=manifest)
    val_ds = BloodCellDataset(split="val", csv_path=manifest)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=(2 if device.type == "cuda" else 0), pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=(2 if device.type == "cuda" else 0), pin_memory=(device.type == "cuda")
    )

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )

    # First experiment: train the new classifier only.
    for param in model.features.parameters():
        param.requires_grad = False

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )

    history = []
    history_path = report_dir / "mobilenet_training_history.csv"
    checkpoint_path = model_dir / "best_mobilenet.pt"
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep the frozen feature extractor in evaluation mode.
        model.features.eval()
        train_loss_sum = 0.0
        train_preds, train_targets = [], []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_preds.extend(logits.argmax(1).detach().cpu().tolist())
            train_targets.extend(labels.detach().cpu().tolist())

        model.eval()
        val_loss_sum = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

        train_loss = train_loss_sum / len(train_ds)
        val_loss = val_loss_sum / len(val_ds)
        train_acc = accuracy_score(train_targets, train_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        train_f1 = f1_score(
            train_targets, train_preds, average="macro", zero_division=0
        )
        val_f1 = f1_score(
            val_targets, val_preds, average="macro", zero_division=0
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"train_macro_f1={train_f1:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "image_size": 224,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "model_name": "MobileNetV2",
                "pretrained": True,
                "frozen_features": True,
            }, checkpoint_path)
            print("  Saved new best checkpoint.")

    print("\nTraining complete.")
    print("History:", history_path)
    print("Best checkpoint:", checkpoint_path)
    print(f"Best validation macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

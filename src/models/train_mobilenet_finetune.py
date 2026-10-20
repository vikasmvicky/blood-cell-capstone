"""Fine-tune ImageNet-pretrained MobileNetV2 on the blood-cell dataset.

Extracted from Blood_Cell_CNN_Training.ipynb cell 67 -- the run that produced
outputs/models/best_mobilenet_finetuned.pt (best validation macro F1 0.9754),
outputs/reports/mobilenet_finetuned_training_history.csv and
outputs/reports/mobilenet_finetuned_summary.json.
Local adaptations: project root from __file__, CUDA-only AMP, CLI knobs
(defaults match the notebook run), splits.csv manifest via the dataset loader.
"""

import argparse
import json
import time
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from src.data.image_dataset import BloodCellDataset, load_class_mapping

PROJECT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=64)
    args = parser.parse_args()






    # Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Save all experiment artifacts in the project
    MODEL_DIR = PROJECT / "outputs/models"
    REPORT_DIR = PROJECT / "outputs/reports"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ImageNet normalization for pretrained MobileNetV2
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Existing split CSV and dataset loader
    train_ds = BloodCellDataset("train")
    val_ds = BloodCellDataset("val")

    # Windows CPU DataLoader workers can segfault; keep workers for GPU runs.
    num_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds, batch_size=args.train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    class_to_idx = load_class_mapping()
    num_classes = len(class_to_idx)
    print("Classes:", class_to_idx)
    print("Train:", len(train_ds), "| Validation:", len(val_ds))

    # Class-weighted cross-entropy, based on training split counts
    train_labels = train_ds.df["class_name"].map(class_to_idx)
    counts = np.bincount(train_labels.to_numpy(), minlength=num_classes)
    weights = len(train_labels) / (num_classes * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # Load ImageNet-pretrained MobileNetV2
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )
    model.classifier[1] = nn.Linear(
        model.last_channel, num_classes
    )

    # Freeze earlier feature layers; fine-tune the final blocks
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[14:].parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam([
        {"params": model.features[14:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        train_losses, train_true, train_pred = [], [], []

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs} — train", leave=False
        ):
            images = normalize(images).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_true.extend(labels.cpu().tolist())
            train_pred.extend(logits.argmax(1).detach().cpu().tolist())

        model.eval()
        val_losses, val_true, val_pred = [], [], []

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch}/{args.epochs} — val", leave=False
            ):
                images = normalize(images).to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_losses.append(loss.item())
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())

        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(
            train_true, train_pred, average="macro", zero_division=0
        )
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(
            val_true, val_pred, average="macro", zero_division=0
        )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": float(np.mean(val_losses)),
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "epoch_seconds": time.time() - start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            REPORT_DIR / "mobilenet_finetuned_training_history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train acc {train_acc:.4f} | train F1 {train_f1:.4f} | "
            f"val acc {val_acc:.4f} | val macro F1 {val_f1:.4f} | "
            f"{row['epoch_seconds']:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

            torch.save({
                "model_state_dict": best_state,
                "class_to_idx": class_to_idx,
                "image_size": [224, 224],
                "epoch": best_epoch,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "model_name": "MobileNetV2",
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            }, MODEL_DIR / "best_mobilenet_finetuned.pt")

    # Record the validation-selected checkpoint summary
    summary = {
        "model": "MobileNetV2",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "epochs_completed": len(history),
        "selection_metric": "validation_macro_f1",
        "test_evaluated_during_training": False,
    }
    (REPORT_DIR / "mobilenet_finetuned_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n✅ Fine-tuning finished.")
    print("Best epoch:", best_epoch)
    print("Best validation macro F1:", round(best_val_f1, 4))
    print("Checkpoint:", MODEL_DIR / "best_mobilenet_finetuned.pt")
    print("History:", REPORT_DIR / "mobilenet_finetuned_training_history.csv")


if __name__ == "__main__":
    main()

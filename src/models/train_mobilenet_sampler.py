
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import accuracy_score, f1_score

from src.data.image_dataset import BloodCellDataset


def main():
    root = Path(__file__).resolve().parents[2]
    reports = root / "outputs/reports"
    models_dir = root / "outputs/models"
    manifest = reports / "splits.csv"

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    with open(reports / "class_to_idx.json") as f:
        class_to_idx = json.load(f)

    df = pd.read_csv(manifest)
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    counts = train_df["class_name"].value_counts()
    sample_weights = train_df["class_name"].map(
        lambda name: 1.0 / counts[name]
    ).to_numpy()

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_ds = BloodCellDataset(split="train", csv_path=manifest)
    val_ds = BloodCellDataset(split="val", csv_path=manifest)

    train_loader = DataLoader(
        train_ds, batch_size=32, sampler=sampler,
        num_workers=(2 if device.type == "cuda" else 0), pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=(2 if device.type == "cuda" else 0), pin_memory=(device.type == "cuda")
    )

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(class_to_idx)
    )

    # Fine-tune the same final feature blocks as before.
    for param in model.features.parameters():
        param.requires_grad = False
    for layer in model.features[14:]:
        for param in layer.parameters():
            param.requires_grad = True

    model = model.to(device)

    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)

    # Ordinary loss: no class weights in this experiment.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {
            "params": [
                p for p in model.features.parameters() if p.requires_grad
            ],
            "lr": 1e-4,
        },
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ])

    history = []
    best_f1 = -1.0
    checkpoint_path = models_dir / "best_mobilenet_sampler.pt"

    for epoch in range(1, 6):
        model.train()
        model.features[:14].eval()

        train_loss_sum = 0.0
        train_true, train_pred = [], []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = (images - mean) / std

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_true.extend(labels.cpu().tolist())
            train_pred.extend(logits.argmax(1).detach().cpu().tolist())

        model.eval()
        val_loss_sum = 0.0
        val_true, val_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                images = (images - mean) / std
                logits = model(images)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())

        train_loss = train_loss_sum / len(train_ds)
        val_loss = val_loss_sum / len(val_ds)
        train_acc = accuracy_score(train_true, train_pred)
        val_acc = accuracy_score(val_true, val_pred)
        train_f1 = f1_score(
            train_true, train_pred, average="macro", zero_division=0
        )
        val_f1 = f1_score(
            val_true, val_pred, average="macro", zero_division=0
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
        pd.DataFrame(history).to_csv(
            reports / "mobilenet_sampler_history.csv", index=False
        )

        print(
            f"Epoch {epoch}/5 | train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | train_macro_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "model_name": "MobileNetV2_WeightedSampler",
                "imagenet_normalization": True,
            }, checkpoint_path)
            print("  Saved best sampler checkpoint.")

    print("\nBest validation macro F1:", round(best_f1, 4))
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()

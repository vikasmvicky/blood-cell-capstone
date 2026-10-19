
"""Train and validate the blood-cell CNN."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.image_dataset import BloodCellDataset
from src.models.cnn_model import BloodCellCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def run_epoch(model, loader, criterion, device, optimizer=None,
              max_batches=None):
    """Run training when an optimizer is supplied; otherwise validate."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No samples were processed.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the blood-cell CNN."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    train_dataset = BloodCellDataset("train")
    val_dataset = BloodCellDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    class_to_idx = train_dataset.class_to_idx
    model = BloodCellCNN(num_classes=len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = OUTPUT_DIR / "best_cnn.pt"
    history_path = REPORT_DIR / "cnn_training_history.csv"
    history = []
    best_val_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "train_samples": train_metrics["samples"],
            "val_samples": val_metrics["samples"],
            "seconds": round(time.time() - start, 2),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(
            f"Train loss: {row['train_loss']:.4f} | "
            f"Train accuracy: {row['train_accuracy']:.2%} "
            f"({row['train_samples']} images)"
        )
        print(
            f"Val loss:   {row['val_loss']:.4f} | "
            f"Val accuracy: {row['val_accuracy']:.2%} "
            f"({row['val_samples']} images)"
        )
        print(f"Time: {row['seconds']:.2f} seconds")

        # Save the best checkpoint based on validation accuracy.
        if row["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = row["val_accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "image_size": [224, 224],
                    "epoch": epoch,
                    "val_accuracy": best_val_accuracy,
                    "model_name": "BloodCellCNN",
                },
                checkpoint_path,
            )
            print("Best model checkpoint saved.")

    print("\nTraining run completed.")
    print("History:", history_path)
    print("Checkpoint:", checkpoint_path)
    print(
        "Note: validation accuracy from a limited-batch run "
        "is only a smoke-test metric."
        if args.max_val_batches is not None
        else "Validation accuracy is based on the complete validation set."
    )


if __name__ == "__main__":
    main()

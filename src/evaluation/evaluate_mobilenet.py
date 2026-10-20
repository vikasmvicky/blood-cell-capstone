"""Evaluate the fine-tuned MobileNetV2 checkpoint on the held-out test split.

Extracted from Blood_Cell_CNN_Training.ipynb cells 69/70 (identical).
Writes mobilenet_finetuned_test_report.csv,
mobilenet_finetuned_confusion_matrix.csv and
mobilenet_finetuned_test_metrics.json to outputs/reports/.
Run from the project root:  python -m src.evaluation.evaluate_mobilenet
"""

def main():

    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix,
    )

    from src.data.image_dataset import BloodCellDataset

    PROJECT = Path(__file__).resolve().parents[2]
    MODEL_PATH = PROJECT / "outputs/models/best_mobilenet_finetuned.pt"
    REPORT_DIR = PROJECT / "outputs/reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluation device:", device)

    # Load the saved checkpoint
    checkpoint = torch.load(
        MODEL_PATH, map_location="cpu", weights_only=True
    )

    class_to_idx = checkpoint["class_to_idx"]
    class_names = [
        name for name, idx in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]

    # Rebuild the same MobileNetV2 architecture
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel, len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ImageNet normalization used during training
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)

    # Use the held-out test split; no training or augmentation here
    test_ds = BloodCellDataset("test")
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    all_labels = []
    all_preds = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = (images - mean) / std
            logits = model(images)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    # Calculate final test metrics
    accuracy = accuracy_score(all_labels, all_preds)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(len(class_names)))
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{name}" for name in class_names],
        columns=[f"predicted_{name}" for name in class_names]
    )

    metrics = {
        "model": "MobileNetV2 fine-tuned",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_samples": len(all_labels),
        "num_classes": len(class_names),
        "class_names": class_names,
        "test_accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }

    report_df.to_csv(
        REPORT_DIR / "mobilenet_finetuned_test_report.csv"
    )
    cm_df.to_csv(
        REPORT_DIR / "mobilenet_finetuned_confusion_matrix.csv"
    )
    with open(
        REPORT_DIR / "mobilenet_finetuned_test_metrics.json", "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nFINAL TEST RESULTS")
    print(f"Test samples:      {len(all_labels)}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro precision:   {macro_p:.4f}")
    print(f"Macro recall:      {macro_r:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")

    print("\nPER-CLASS REPORT")
    print(report_df.to_string())

    print("\nSaved reports:")
    print(REPORT_DIR / "mobilenet_finetuned_test_report.csv")
    print(REPORT_DIR / "mobilenet_finetuned_confusion_matrix.csv")
    print(REPORT_DIR / "mobilenet_finetuned_test_metrics.json")


if __name__ == "__main__":
    main()

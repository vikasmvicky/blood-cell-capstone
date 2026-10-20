"""Classification metrics, confusion matrix and error analysis (torch-free)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> dict:
    """Overall, macro/weighted and per-class metrics. Keys match the legacy evaluate_mobilenet.py."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length.")
    labels = list(range(len(class_names)))

    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    per_class = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names), output_dict=True, zero_division=0
    )
    return {
        "test_samples": int(y_true.size),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "weighted_precision": float(wp),
        "weighted_recall": float(wr),
        "weighted_f1": float(wf),
        "per_class": {k: v for k, v in per_class.items() if k in class_names},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def top_confusions(cm: Sequence[Sequence[int]], class_names: Sequence[str], k: int = 10) -> pd.DataFrame:
    """Most frequent off-diagonal (true -> predicted) confusions with row-normalised rates."""
    cm = np.asarray(cm)
    rows = []
    for i in range(cm.shape[0]):
        support = cm[i].sum()
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "true_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(cm[i, j]),
                        "rate_of_true_class": float(cm[i, j] / support) if support else 0.0,
                    }
                )
    df = pd.DataFrame(rows, columns=["true_class", "predicted_class", "count", "rate_of_true_class"])
    return df.sort_values(["count", "rate_of_true_class"], ascending=False).head(k).reset_index(drop=True)


def per_class_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics["per_class"]).T[["precision", "recall", "f1-score", "support"]]


def plot_confusion_matrix(cm: Sequence[Sequence[int]], class_names: Sequence[str], path, normalize: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if shown[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

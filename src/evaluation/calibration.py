"""Probability-calibration utilities (numpy only).

Softmax outputs are *confidence scores*, not guaranteed probabilities. These
helpers measure calibration (Expected Calibration Error, NLL) and fit a single
temperature on the **validation** split. Temperature scaling never changes the
arg-max prediction, so accuracy is unchanged; it only rescales probabilities.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Standard equal-width-bin ECE on top-1 confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between probabilities and one-hot labels.

    Lower is better; a uniform 1/8 prediction over 8 classes scores 7/8 = 0.875.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Per-bin confidence/accuracy counts for a reliability diagram."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "bin_center": float((lo + hi) / 2),
                "count": int(in_bin.sum()),
                "mean_confidence": float(conf[in_bin].mean()) if in_bin.any() else None,
                "accuracy": float(correct[in_bin].mean()) if in_bin.any() else None,
            }
        )
    return rows


def plot_reliability_diagram(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    path,
    n_bins: int = 15,
    title_before: str = "raw softmax",
    title_after: str = "temperature-scaled",
) -> None:
    """One figure: reliability curves before/after calibration against perfect calibration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for probs, label, color in (
        (probs_before, f"{title_before} (ECE {expected_calibration_error(probs_before, labels, n_bins):.4f})", "#d62728"),
        (probs_after, f"{title_after} (ECE {expected_calibration_error(probs_after, labels, n_bins):.4f})", "#1f77b4"),
    ):
        rows = [r for r in reliability_table(probs, labels, n_bins) if r["count"] > 0]
        xs = [r["mean_confidence"] for r in rows]
        ys = [r["accuracy"] for r in rows]
        sizes = [max(18.0, 2.2 * r["count"]) for r in rows]
        ax.plot(xs, ys, "o-", color=color, label=label)
        for r, x, y, s in zip(rows, xs, ys, sizes):
            ax.scatter(x, y, s=s, color=color, alpha=0.35)
    ax.set_xlabel("Confidence (softmax probability of predicted class)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.005)
    ax.set_title("Reliability diagram")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Pick T minimising validation NLL by a deterministic coarse-to-fine grid search."""
    labels = np.asarray(labels)
    grid = np.linspace(0.25, 5.0, 96) if grid is None else grid
    best_t = min(grid, key=lambda t: negative_log_likelihood(softmax(logits, t), labels))
    fine = np.linspace(max(0.05, best_t - 0.05), best_t + 0.05, 101)
    return float(min(fine, key=lambda t: negative_log_likelihood(softmax(logits, t), labels)))

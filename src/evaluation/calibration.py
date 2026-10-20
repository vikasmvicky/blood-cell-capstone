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


def fit_temperature(logits: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Pick T minimising validation NLL by a deterministic coarse-to-fine grid search."""
    labels = np.asarray(labels)
    grid = np.linspace(0.25, 5.0, 96) if grid is None else grid
    best_t = min(grid, key=lambda t: negative_log_likelihood(softmax(logits, t), labels))
    fine = np.linspace(max(0.05, best_t - 0.05), best_t + 0.05, 101)
    return float(min(fine, key=lambda t: negative_log_likelihood(softmax(logits, t), labels)))

"""Tests for Brier score, reliability tables and the reliability-diagram plot."""

import numpy as np
import pytest

from src.evaluation import calibration as cal


def test_brier_score_perfect_and_uniform():
    labels = np.array([0, 1, 2])
    perfect = np.eye(3)[labels]
    assert cal.brier_score(perfect, labels) == pytest.approx(0.0)
    uniform = np.full((3, 3), 1 / 3)
    # per sample: (1/3-1)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 2/3
    assert cal.brier_score(uniform, labels) == pytest.approx(2 / 3)
    assert cal.brier_score(uniform, labels) < cal.brier_score(np.eye(3)[(labels + 1) % 3], labels)


def test_reliability_table_counts_and_ranges():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 4))
    labels = rng.integers(0, 4, size=200)
    rows = cal.reliability_table(cal.softmax(logits), labels, n_bins=10)
    assert len(rows) == 10
    assert sum(r["count"] for r in rows) == 200
    filled = [r for r in rows if r["count"]]
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in filled)
    assert all(r["bin_low"] < r["mean_confidence"] < r["bin_high"] for r in filled)


def test_reliability_diagram_written(tmp_path):
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(120, 3)) * 3
    labels = rng.integers(0, 3, size=120)
    probs = cal.softmax(logits)
    out = tmp_path / "reliability.png"
    cal.plot_reliability_diagram(probs, cal.softmax(logits, 2.0), labels, out)
    assert out.stat().st_size > 1000

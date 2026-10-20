"""End-to-end plumbing of the evaluation/benchmark CLIs on synthetic assets.

Random weights => the printed accuracy is meaningless; only artefact creation,
metric consistency and error handling are asserted.
"""

import json

import dataclasses

import numpy as np
import pytest

from src.evaluation import benchmark, run_evaluation


@pytest.fixture()
def patched_config(synthetic_checkpoint, monkeypatch, tmp_path):
    cfg = synthetic_checkpoint
    monkeypatch.setattr(run_evaluation, "load_config", lambda path=None: cfg)
    return cfg


def test_run_evaluation_writes_reports(patched_config, tmp_path):
    out = tmp_path / "reports"
    assert run_evaluation.main(["--report-dir", str(out), "--calibrate", "--batch-size", "8"]) == 0
    metrics = json.loads((out / "mobilenet_finetuned_test_metrics.json").read_text())
    assert metrics["test_samples"] == 16 and metrics["num_classes"] == 8
    cm = np.array(metrics["confusion_matrix"])
    assert cm.sum() == 16 and 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["test_accuracy"] == pytest.approx(np.trace(cm) / cm.sum())
    for name in ("confusion_matrix.png", "top_confusions.csv", "test_predictions.csv", "calibration.json", "mobilenet_finetuned_test_report.csv"):
        assert (out / name).is_file(), name
    cal = json.loads((out / "calibration.json").read_text())
    assert cal["fit_split"] == "val" and len(cal["checkpoint_sha256"]) == 64
    assert "D:" not in (out / "test_predictions.csv").read_text()


def test_run_evaluation_reports_missing_checkpoint(synthetic_assets, monkeypatch, capsys):
    monkeypatch.setattr(run_evaluation, "load_config", lambda path=None: synthetic_assets)
    assert run_evaluation.main([]) == 2
    assert "evaluation unavailable" in capsys.readouterr().err


def test_run_evaluation_reports_missing_images(synthetic_checkpoint, monkeypatch, tmp_path, capsys):
    cfg = dataclasses.replace(synthetic_checkpoint, data_root=tmp_path / "empty")
    monkeypatch.setattr(run_evaluation, "load_config", lambda path=None: cfg)
    assert run_evaluation.main(["--report-dir", str(tmp_path / "r")]) == 2
    assert "ASSETS.md" in capsys.readouterr().err


def test_benchmark_architecture_only(synthetic_assets, monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "load_config", lambda path=None: synthetic_assets)
    out = tmp_path / "b.json"
    assert benchmark.main(["--architecture-only", "--repeats", "3", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["model_source"].startswith("architecture-only") and 2.2e6 < r["parameters"] < 2.25e6
    assert r["latency_ms_batch1"]["mean"] > 0 and r["throughput_images_per_s"] > 0


def test_benchmark_without_checkpoint_fails_cleanly(synthetic_assets, monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "load_config", lambda path=None: synthetic_assets)
    assert benchmark.main([]) == 2
    assert "benchmark unavailable" in capsys.readouterr().err

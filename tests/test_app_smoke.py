"""Streamlit smoke tests using Streamlit's AppTest harness (no browser needed).

The app is a six-section research application (sidebar radio navigation). These
tests exercise every section with: no checkpoint, a valid synthetic checkpoint
and a corrupt checkpoint, plus the missing/unavailable report fallbacks.
"""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import ROOT  # noqa: E402

APP = str(ROOT / "app.py")

SECTIONS = [
    "Research Overview",
    "Prediction Studio",
    "Explainability",
    "Evaluation Dashboard",
    "Calibration",
    "Methodology",
]


def run_app(monkeypatch, model_path, section=None):
    monkeypatch.setenv("BLOOD_CELL_MODEL_PATH", str(model_path))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    if section is not None:
        at.radio[0].set_value(section)
        at.run()
    return at


# --------------------------------------------------------------------------- #
# Checkpoint handling on the landing page (Research Overview)
# --------------------------------------------------------------------------- #

def test_app_without_checkpoint_shows_setup_message(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path / "missing.pt")
    assert not at.exception
    assert any("checkpoint not found" in e.value.lower() for e in at.error)
    assert any("research and educational use only" in w.value.lower() for w in at.warning)


def test_app_with_checkpoint_loads_on_overview(monkeypatch, synthetic_checkpoint):
    at = run_app(monkeypatch, synthetic_checkpoint.model_path)
    assert not at.exception
    assert not at.error
    assert any("checkpoint loaded" in m.value.lower() for m in at.markdown)


def test_app_with_incompatible_checkpoint_shows_error(monkeypatch, tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"junk")
    at = run_app(monkeypatch, bad)
    assert not at.exception
    assert any("could not be loaded" in e.value.lower() for e in at.error)


# --------------------------------------------------------------------------- #
# Every section renders without exception (with a valid checkpoint)
# --------------------------------------------------------------------------- #

def test_all_sections_render_with_checkpoint(monkeypatch, synthetic_checkpoint):
    at = run_app(monkeypatch, synthetic_checkpoint.model_path)
    assert not at.exception
    for section in SECTIONS:
        at.radio[0].set_value(section)
        at.run()
        assert not at.exception, f"exception in section '{section}'"


def test_prediction_studio_waits_for_upload(monkeypatch, synthetic_checkpoint):
    at = run_app(monkeypatch, synthetic_checkpoint.model_path, section="Prediction Studio")
    assert not at.exception
    assert not at.error
    assert any("upload an image" in i.value.lower() for i in at.info)


def test_prediction_studio_without_checkpoint_errors(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path / "missing.pt", section="Prediction Studio")
    assert not at.exception
    assert any("checkpoint not found" in e.value.lower() for e in at.error)


def test_explainability_without_upload_asks_for_image(monkeypatch, synthetic_checkpoint):
    at = run_app(monkeypatch, synthetic_checkpoint.model_path, section="Explainability")
    assert not at.exception
    assert any("upload an image" in i.value.lower() for i in at.info)


def test_sections_are_graceful_without_reports(monkeypatch, synthetic_checkpoint):
    """Report-dependent sections must show informative fallbacks, never exceptions."""
    at = run_app(monkeypatch, synthetic_checkpoint.model_path)
    for section in ("Evaluation Dashboard", "Calibration", "Methodology"):
        at.radio[0].set_value(section)
        at.run()
        assert not at.exception, section

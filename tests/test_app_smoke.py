"""Streamlit smoke tests using Streamlit's AppTest harness (no browser needed)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import ROOT  # noqa: E402

APP = str(ROOT / "app.py")


def run_app(monkeypatch, model_path):
    monkeypatch.setenv("BLOOD_CELL_MODEL_PATH", str(model_path))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_without_checkpoint_shows_setup_message(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path / "missing.pt")
    assert not at.exception
    assert any("checkpoint not found" in e.value.lower() for e in at.error)
    assert any("research and educational use only" in w.value.lower() for w in at.warning)


def test_app_with_checkpoint_loads_and_waits_for_upload(monkeypatch, synthetic_checkpoint):
    at = run_app(monkeypatch, synthetic_checkpoint.model_path)
    assert not at.exception
    assert not at.error
    assert any("upload an image" in i.value.lower() for i in at.info)
    assert at.title[0].value.endswith("Explainable Blood Cell Classification")


def test_app_with_incompatible_checkpoint_shows_error(monkeypatch, tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"junk")
    at = run_app(monkeypatch, bad)
    assert not at.exception
    assert any("could not be loaded" in e.value.lower() for e in at.error)

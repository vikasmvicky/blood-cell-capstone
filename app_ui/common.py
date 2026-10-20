"""Shared UI helpers: design system, cached model/report loading, small components."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import PROJECT_ROOT, Config

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

DISCLAIMER = (
    "**Research and educational use only.** This tool is not a medical device and its output is not a "
    "clinical diagnosis. Blood smears must be reviewed by qualified laboratory professionals."
)

CSS = """
<style>
  .bca-kpi { background:#f5f7fb; border:1px solid #e3e8f0; border-radius:10px; padding:10px 14px; }
  .bca-kpi .label { color:#5a6478; font-size:0.78rem; text-transform:uppercase; letter-spacing:.04em; }
  .bca-kpi .value { color:#1f2937; font-size:1.35rem; font-weight:700; }
  .bca-chip { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; margin-right:6px;}
  .bca-chip.ok   { background:#e5f6ec; color:#136c3b; }
  .bca-chip.warn { background:#fdf1d8; color:#8a5b00; }
  .bca-chip.bad  { background:#fbe3e3; color:#8f1d1d; }
  .bca-note { color:#5a6478; font-size:.85rem; }
</style>
"""


def apply_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def footer() -> None:
    st.markdown("---")
    st.caption(
        "Blood-cell classification research prototype · PBC dataset · MobileNetV2 + Grad-CAM · "
        "not for clinical use."
    )


# --------------------------------------------------------------------------- #
# Cached model + artifact loading
# --------------------------------------------------------------------------- #

def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


@st.cache_resource(show_spinner="Loading model checkpoint...")
def _load_predictor_cached(model_path: str, model_mtime: float, cal_mtime: float):
    from src.inference.service import build_predictor

    return build_predictor(load_config_imported())


def load_config_imported() -> Config:
    from src.config import load_config

    return load_config()


def get_predictor(cfg: Config):
    """Return (predictor, error). Never substitutes an untrained model."""
    from src.inference.errors import CheckpointError

    try:
        predictor = _load_predictor_cached(
            str(cfg.model_path), _mtime(cfg.model_path), _mtime(REPORTS_DIR / "calibration.json")
        )
        return predictor, None
    except CheckpointError as exc:
        return None, exc
    except ImportError as exc:
        return None, exc
    except Exception as exc:  # noqa: BLE001 - surface anything as a load error
        return None, exc


def render_checkpoint_error(cfg: Config, exc: Exception) -> None:
    from src.inference.errors import CheckpointNotFoundError

    if isinstance(exc, CheckpointNotFoundError):
        st.error("**Model checkpoint not found - predictions and explanations are unavailable.**")
        st.markdown(
            f"Place the trained file `best_mobilenet_finetuned.pt` at:\n\n`{cfg.model_path}`\n\n"
            "or set `BLOOD_CELL_MODEL_PATH`. See `ASSETS.md`. No substitute or randomly "
            "initialised model is ever used."
        )
    else:
        st.error(f"**The model could not be loaded:** {exc}")


@st.cache_data(show_spinner=False)
def read_json_artifact(name: str, mtime: float):
    path = REPORTS_DIR / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


@st.cache_data(show_spinner=False)
def read_csv_artifact(name: str, mtime: float):
    path = REPORTS_DIR / name
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (ValueError, OSError):
        return None


def load_report_json(name: str):
    return read_json_artifact(name, _mtime(REPORTS_DIR / name))


def load_report_csv(name: str):
    return read_csv_artifact(name, _mtime(REPORTS_DIR / name))


def report_exists(name: str) -> bool:
    return (REPORTS_DIR / name).is_file()


def download_bytes(data: bytes | str, filename: str, label: str, mime: str = "text/plain") -> None:
    if isinstance(data, str):
        data = data.encode("utf-8")
    st.download_button(label=label, data=data, file_name=filename, mime=mime)


def download_json(obj, filename: str, label: str) -> None:
    download_bytes(json.dumps(obj, indent=2), filename, label, "application/json")


def kpi(label: str, value: str, note: str = "") -> None:
    note_html = f'<div class="bca-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="bca-kpi"><div class="label">{label}</div><div class="value">{value}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def status_chip(kind: str, text: str) -> str:
    return f'<span class="bca-chip {kind}">{text}</span>'


def pct(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"

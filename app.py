"""Streamlit research application: Explainable AI-Based Blood Cell Classification.

Run:  streamlit run app.py

Six sections (sidebar navigation):

1. Research Overview      - project, dataset, model, verified results, limitations
2. Prediction Studio      - upload one image, probabilities, uncertainty, export
3. Explainability         - Grad-CAM comparison workspace with opacity control
4. Evaluation Dashboard   - held-out test metrics, confusion matrix, error analysis
5. Calibration            - ECE/NLL/Brier, reliability diagram, temperature scaling
6. Methodology            - dataset, training, evaluation and experiment protocols

Research/educational use only - NOT a medical device and NOT a clinical diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

st.set_page_config(
    page_title="Blood Cell Classifier (research demo)",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app_ui import calibration_page, common, evaluation_page, explainability, methodology, overview, prediction  # noqa: E402

SECTIONS = {
    "Research Overview": overview.render,
    "Prediction Studio": prediction.render,
    "Explainability": explainability.render,
    "Evaluation Dashboard": evaluation_page.render,
    "Calibration": calibration_page.render,
    "Methodology": methodology.render,
}


def main() -> None:
    common.apply_styles()
    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 - configuration problems must be user-visible
        st.error(f"Configuration error: {exc}")
        st.stop()

    with st.sidebar:
        st.title("\U0001f52c Blood Cell AI")
        st.caption("Explainable research prototype")
        section = st.radio("Section", list(SECTIONS), label_visibility="collapsed")
        st.markdown("---")
        st.caption(common.DISCLAIMER)

    render = SECTIONS[section]
    render(cfg)
    common.footer()


main()

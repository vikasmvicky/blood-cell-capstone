"""Streamlit website: Explainable AI-Based Blood Cell Classification.

Run:  streamlit run app.py
Research/educational use only - NOT a medical device and NOT a clinical diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.inference.errors import CheckpointError, CheckpointNotFoundError, InvalidImageError  # noqa: E402

st.set_page_config(page_title="Blood Cell Classifier (research demo)", page_icon="🔬", layout="wide")

DISCLAIMER = (
    "**Research and educational use only.** This tool is not a medical device and its output is not a "
    "clinical diagnosis. Blood smears must be reviewed by qualified laboratory professionals."
)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_path: str, mtime: float):
    """Cached per (path, modification time) so a replaced checkpoint is picked up."""
    from src.inference.service import build_predictor

    return build_predictor(load_config())


def render_header() -> None:
    st.title("🔬 Explainable Blood Cell Classification")
    st.caption("MobileNetV2 + Grad-CAM on the eight-class Peripheral Blood Cell (PBC) dataset")
    st.warning(DISCLAIMER, icon="⚠️")


def render_project_info(cfg) -> None:
    with st.expander("About this project, supported classes and limitations", expanded=False):
        st.markdown(
            "**Classes (8):** " + ", ".join(cfg.display_name(c) for c in cfg.display_names) + ".\n\n"
            "**How it works.** The image is resized to 224×224, normalised with ImageNet statistics and passed to a "
            "fine-tuned MobileNetV2. Grad-CAM highlights image regions that increased the score of the predicted class.\n\n"
            "**Limitations**\n"
            "- Trained and evaluated only on one dataset (PBC, single acquisition setup). Other microscopes, stains or "
            "labs may behave very differently.\n"
            "- The model always outputs one of the eight classes, even for images that are not blood cells, or "
            "that show other cell types, artefacts or several overlapping cells.\n"
            "- The *confidence* is the model's softmax score. It is **not** a calibrated probability and **not** a "
            "measure of medical certainty (a calibrated version is shown only if a temperature was fitted for this exact checkpoint).\n"
            "- Grad-CAM maps are coarse (7×7 grid upsampled) and qualitative; they do not prove the model reasons like a haematologist.\n"
            "- Best results: a single, centred, stained white-cell/platelet/erythroblast crop similar to PBC images."
        )


def render_setup_help(exc: Exception, cfg) -> None:
    if isinstance(exc, CheckpointNotFoundError):
        st.error("**Model checkpoint not found - predictions are unavailable.**")
        st.markdown(
            f"Place the trained file `best_mobilenet_finetuned.pt` at:\n\n`{cfg.model_path}`\n\n"
            "or set the environment variable `BLOOD_CELL_MODEL_PATH`. Details are in `ASSETS.md`. "
            "No substitute or untrained model is used."
        )
    else:
        st.error(f"**The model could not be loaded:** {exc}")


def render_prediction(image, pred, cfg, show_cam: bool) -> None:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)
        st.caption(f"{image.width}×{image.height}px (resized to {cfg.image_size[0]}×{cfg.image_size[1]} for the model)")
    with right:
        st.subheader("Prediction")
        st.metric("Predicted class", pred.display_name)
        st.metric(f"Model confidence ({pred.probability_kind})", f"{pred.confidence * 100:.1f}%")
        st.caption(
            f"Margin over 2nd class: {pred.margin * 100:.1f} pts · normalised entropy: {pred.normalized_entropy:.2f}. "
            "Confidence is a model score, not a medical certainty."
        )
        if pred.low_confidence:
            st.warning(f"Confidence is below {cfg.low_confidence_threshold * 100:.0f}%: the model is uncertain; treat this result with extra caution.")
        for w in pred.quality_warnings:
            st.warning(w.message + " (heuristic check)")

    st.subheader("Class probabilities")
    df = pd.DataFrame(
        {"Class": [cfg.display_name(n) for n, _ in pred.ranked], "Probability": [p for _, p in pred.ranked]}
    )
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={"Probability": st.column_config.ProgressColumn("Probability", min_value=0.0, max_value=1.0, format="percent")},
    )

    if show_cam:
        st.subheader("Explanation (Grad-CAM)")
        if pred.overlay is not None:
            c1, c2 = st.columns(2)
            c1.image(image.resize(cfg.image_size), caption="Model input", use_container_width=True)
            c2.image(pred.overlay, caption=f"Grad-CAM for '{pred.display_name}' (red = stronger influence)", use_container_width=True)
            st.info(
                "Warm regions contributed most to the score of the predicted class. The map is coarse and qualitative; "
                "if it highlights background instead of the cell, be sceptical of the prediction."
            )
        else:
            st.info(pred.gradcam_error or "Grad-CAM is not available.")


def main() -> None:
    render_header()
    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Configuration error: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        show_cam = st.toggle("Show Grad-CAM explanation", value=True)
        st.markdown("---")
        st.caption("Accepted: PNG, JPG, JPEG · up to %d MB" % cfg.max_upload_mb)

    predictor, load_error = None, None
    try:
        mtime = cfg.model_path.stat().st_mtime if cfg.model_path.exists() else 0.0
        predictor = get_predictor(str(cfg.model_path), mtime)
    except CheckpointError as exc:
        load_error = exc
    except ImportError as exc:
        load_error = exc
        st.error(f"A required dependency is missing: {exc}. Run `pip install -r requirements.txt`.")
    except Exception as exc:  # noqa: BLE001
        load_error = exc

    st.header("1 · Upload")
    upload = st.file_uploader("Choose a blood-cell image", type=list(cfg.allowed_extensions), help="Single stained cell, similar to PBC images.")

    if load_error is not None and not isinstance(load_error, ImportError):
        render_setup_help(load_error, cfg)

    if upload is not None:
        from src.inference.service import validate_upload

        data = upload.getvalue()
        try:
            image = validate_upload(data, upload.name, cfg)
        except InvalidImageError as exc:
            st.error(f"**Cannot use this file:** {exc}")
        else:
            if predictor is None:
                st.image(image, caption="Uploaded image (no prediction without a model)", width=360)
            else:
                st.header("2 · Prediction and explanation")
                with st.spinner("Classifying..."):
                    pred = predictor.predict(image, with_gradcam=show_cam)
                render_prediction(image, pred, cfg, show_cam)
                if predictor.loaded.notes:
                    st.caption(" ".join(predictor.loaded.notes))
    else:
        st.info("Upload an image to see a prediction.")

    st.header("3 · Project information")
    render_project_info(cfg)
    st.caption(DISCLAIMER)


main()

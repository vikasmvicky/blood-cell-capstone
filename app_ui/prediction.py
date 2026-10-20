"""Section B: Prediction Studio - single-image upload, probabilities, uncertainty, export."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from app_ui import common
from src.config import Config


def _store_session(image: Image.Image, pred, cfg: Config) -> None:
    """Remember the upload so the Explainability section can reuse it."""
    st.session_state["studio_image"] = image
    st.session_state["studio_pred"] = {
        "class_name": pred.class_name,
        "display_name": pred.display_name,
        "confidence": pred.confidence,
        "probability_kind": pred.probability_kind,
    }
    st.session_state["studio_ranked"] = [
        {"class": n, "display": cfg.display_name(n), "probability": p} for n, p in pred.ranked
    ]


def _export_rows(pred, cfg: Config) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class": [n for n, _ in pred.ranked],
            "display_name": [cfg.display_name(n) for n, _ in pred.ranked],
            "probability": [p for _, p in pred.ranked],
        }
    )


def _render_prediction(image: Image.Image, pred, cfg: Config) -> None:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Uploaded image")
        st.image(image, width="stretch")
        st.caption(
            f"{image.width}×{image.height}px · resized to {cfg.image_size[0]}×{cfg.image_size[1]} for the model"
        )
    with right:
        st.subheader("Prediction")
        st.metric("Predicted class", pred.display_name)
        st.metric(f"Confidence ({pred.probability_kind})", common.pct(pred.confidence, 1))
        m1, m2 = st.columns(2)
        m1.metric("Top-2 margin", common.pct(pred.margin, 1))
        m2.metric("Normalised entropy", f"{pred.normalized_entropy:.2f}")
        st.caption(
            "Margin = gap between the top two classes (small = ambiguous). Entropy is 0 for a peaked "
            "distribution and 1 for a uniform one. Confidence is a model score, **not** a medical certainty."
        )
        if pred.low_confidence:
            st.warning(
                f"Confidence below {cfg.low_confidence_threshold:.0%}: the model is uncertain - "
                "treat this result with extra caution."
            )
        for w in pred.quality_warnings:
            st.warning(f"{w.message} (heuristic image check, not validated)")

    st.subheader("Class probabilities")
    st.dataframe(
        _export_rows(pred, cfg),
        hide_index=True,
        width="stretch",
        column_config={
            "probability": st.column_config.ProgressColumn(
                "Probability", min_value=0.0, max_value=1.0, format="percent"
            )
        },
    )

    with st.expander("Export prediction"):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "generated_at": now,
            "disclaimer": "Research prototype - not a clinical diagnosis.",
            "model": "MobileNetV2 fine-tuned",
            "probability_kind": pred.probability_kind,
            "predicted_class": pred.class_name,
            "confidence": pred.confidence,
            "margin_top2": pred.margin,
            "normalized_entropy": pred.normalized_entropy,
            "low_confidence": pred.low_confidence,
            "quality_warnings": [w.code for w in pred.quality_warnings],
            "probabilities": {n: p for n, p in pred.ranked},
        }
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download JSON",
            data=json.dumps(payload, indent=2),
            file_name="prediction.json",
            mime="application/json",
            width="stretch",
        )
        c2.download_button(
            "Download CSV",
            data=_export_rows(pred, cfg).to_csv(index=False),
            file_name="prediction.csv",
            mime="text/csv",
            width="stretch",
        )


def render(cfg: Config) -> None:
    st.title("Prediction Studio")
    st.caption("Upload one blood-cell image; the model returns a class, probabilities and an uncertainty summary.")

    predictor, err = common.get_predictor(cfg)
    if predictor is None:
        common.render_checkpoint_error(cfg, err)
        return

    upload = st.file_uploader(
        "Choose a blood-cell image",
        type=list(cfg.allowed_extensions),
        help="A single stained cell crop, similar to the PBC dataset images.",
    )
    if upload is None:
        st.info("Upload an image to see a prediction.")
        st.caption(f"Accepted: {', '.join(cfg.allowed_extensions)} · up to {cfg.max_upload_mb:.0f} MB")
        return

    from src.inference.errors import InvalidImageError
    from src.inference.service import validate_upload

    data = upload.getvalue()
    try:
        image = validate_upload(data, upload.name, cfg)
    except InvalidImageError as exc:
        st.error(f"**Cannot use this file:** {exc}")
        return

    with st.spinner("Classifying..."):
        pred = predictor.predict(image, with_gradcam=False)
    _store_session(image, pred, cfg)
    _render_prediction(image, pred, cfg)

    if predictor.loaded.notes:
        st.caption(" ".join(predictor.loaded.notes))
    st.caption(
        "Next: open **Explainability** to see a Grad-CAM heat-map for this image, or "
        "**Evaluation Dashboard** for the model's measured performance."
    )

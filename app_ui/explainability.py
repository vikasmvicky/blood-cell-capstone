"""Section C: Explainability Workspace - Grad-CAM generation, comparison, opacity control."""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from app_ui import common
from src.config import Config


def _get_image(cfg: Config, predictor) -> Image.Image | None:
    """Prefer the image already uploaded in the Prediction Studio; allow a new upload."""
    studio_image = st.session_state.get("studio_image")
    if studio_image is not None:
        with st.expander("Image source", expanded=False):
            st.caption("Using the image from the Prediction Studio. You can upload a different one below.")
        upload = st.file_uploader("Or upload a different image", type=list(cfg.allowed_extensions))
        if upload is not None:
            from src.inference.errors import InvalidImageError
            from src.inference.service import validate_upload

            try:
                return validate_upload(upload.getvalue(), upload.name, cfg)
            except InvalidImageError as exc:
                st.error(f"**Cannot use this file:** {exc}")
                return None
        return studio_image

    upload = st.file_uploader("Upload a blood-cell image", type=list(cfg.allowed_extensions))
    if upload is None:
        st.info("Upload an image (or use the Prediction Studio first) to generate a Grad-CAM explanation.")
        return None
    from src.inference.errors import InvalidImageError
    from src.inference.service import validate_upload

    try:
        return validate_upload(upload.getvalue(), upload.name, cfg)
    except InvalidImageError as exc:
        st.error(f"**Cannot use this file:** {exc}")
        return None


def _compute_cam(predictor, image: Image.Image, target_idx: int):
    """Return (cam, error). Validates shapes and handles missing gradients gracefully."""
    import torch

    from src.explainability.gradcam import GradCAM

    try:
        x = predictor._input(image)  # (1, 3, H, W), normalised
        with GradCAM(predictor.loaded.model) as engine:
            cam, cls, logits = engine(x, class_idx=target_idx)
        if cam.shape != (x.shape[-2], x.shape[-1]):
            return None, f"Unexpected heat-map shape {cam.shape}; expected {tuple(x.shape[-2:])}."
        if not np.isfinite(cam).all():
            return None, "Heat-map contains non-finite values; the explanation is not usable."
        probs = torch.softmax(logits, dim=1)[0].detach().numpy()
        return (cam, int(cls), float(probs[target_idx])), None
    except RuntimeError as exc:
        return None, f"Grad-CAM could not be computed (missing gradients or unsupported layer): {exc}"
    except Exception as exc:  # noqa: BLE001 - explanation must never crash the app
        return None, f"Grad-CAM unavailable: {exc}"


def render(cfg: Config) -> None:
    st.title("Explainability Workspace")
    st.caption("Grad-CAM highlights the image regions that most increased the score of the selected class.")

    predictor, err = common.get_predictor(cfg)
    if predictor is None:
        common.render_checkpoint_error(cfg, err)
        return

    image = _get_image(cfg, predictor)
    if image is None:
        return

    names = predictor.class_names
    studio_pred = st.session_state.get("studio_pred")
    default_target = studio_pred["class_name"] if studio_pred else names[0]

    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        target_display = st.selectbox(
            "Class to explain",
            [cfg.display_name(n) for n in names],
            index=names.index(default_target),
            help="Defaults to the predicted class from the Prediction Studio.",
        )
        target_idx = names.index(next(n for n in names if cfg.display_name(n) == target_display))
    with ctrl2:
        opacity = st.slider("Heat-map opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

    with st.spinner("Computing Grad-CAM..."):
        result, error = _compute_cam(predictor, image, target_idx)

    if error:
        st.error(error)
        st.info(
            "Grad-CAM needs a differentiable path from the output score to the target convolutional layer. "
            "This model exposes `model.features[-1]` (a 7×7 grid); if the checkpoint architecture changes, "
            "the target layer must be updated."
        )
        return

    cam, cls, prob = result
    from src.explainability.gradcam import overlay_heatmap

    resized = image.resize(predictor.loaded.image_size, Image.Resampling.BILINEAR)
    overlay = overlay_heatmap(resized, cam, alpha=opacity)
    heat_only = Image.fromarray((np.stack([cam] * 3, -1) * 255).astype(np.uint8))

    st.subheader("Comparison")
    c1, c2, c3 = st.columns(3)
    c1.image(resized, caption=f"Model input ({cfg.image_size[0]}×{cfg.image_size[1]})", width="stretch")
    c2.image(heat_only, caption="Grad-CAM (jet; red = strongest positive influence)", width="stretch")
    c3.image(overlay, caption=f"Overlay for '{cfg.display_name(names[cls])}' (p={prob:.2f})", width="stretch")

    st.markdown(
        f"- Explained class: **{cfg.display_name(names[cls])}** · softmax probability **{prob:.2f}**\n"
        f"- Warm (red) regions contributed most to this class score; blue regions contributed little.\n"
        "- The map is computed on the 224×224 model input from a 7×7 feature grid, so it is inherently coarse."
    )
    with st.expander("How to read (and distrust) this map"):
        st.markdown(
            "- A plausible-looking map is **not** proof the model is correct or faithful; see the "
            "quantitative deletion/insertion test in `outputs/reports/gradcam_faithfulness.json`.\n"
            "- If the map highlights background, borders or artifacts rather than the cell, be sceptical.\n"
            "- Different classes on the same image can highlight different regions - that is expected.\n"
            "- Grad-CAM is **not** segmentation: it gives no cell boundary or area.\n"
            "- Method: Grad-CAM, Selvaraju et al., ICCV 2017."
        )

"""Section E: Calibration Workspace - ECE/NLL/Brier, reliability diagram, temperature scaling."""

from __future__ import annotations

import streamlit as st

from app_ui import common
from src.config import Config


def render(cfg: Config) -> None:
    st.title("Calibration Workspace")
    st.caption(
        "Softmax scores are confidence scores, not guaranteed probabilities. Temperature scaling fits a "
        "single parameter on the **validation** split and is then evaluated on the untouched test split."
    )

    cal = common.load_report_json("calibration.json")
    predictor, err = common.get_predictor(cfg)
    if predictor is not None:
        if predictor.temperature:
            st.success(
                f"The running model currently applies temperature **T = {predictor.temperature:.3f}** "
                "(calibration.json matches this checkpoint's sha256)."
            )
        else:
            st.info(
                "No temperature is applied in the app right now: either calibration was not run, or "
                "`calibration.json` was fitted for a different checkpoint."
            )

    if cal is None:
        st.warning(
            "No calibration report found. Generate one with:\n\n"
            "```bash\npython -m src.evaluation.run_evaluation --calibrate\n```"
        )
        return

    split = cal.get("eval_split", "test")
    c = st.columns(5)
    c[0].metric("Temperature T", f"{cal['temperature']:.3f}")
    c[1].metric("ECE before", f"{cal[f'{split}_ece_before']:.4f}")
    c[2].metric("ECE after", f"{cal[f'{split}_ece_after']:.4f}")
    c[3].metric("NLL before", f"{cal[f'{split}_nll_before']:.4f}")
    c[4].metric("NLL after", f"{cal[f'{split}_nll_after']:.4f}")

    c2 = st.columns(3)
    c2[0].metric("Brier before", f"{cal.get(f'{split}_brier_before', float('nan')):.4f}")
    c2[1].metric("Brier after", f"{cal.get(f'{split}_brier_after', float('nan')):.4f}")
    c2[2].metric("Accuracy", common.pct(cal.get(f"{split}_accuracy_after")), "unchanged by construction")

    st.markdown(
        f"- Temperature was fitted by minimising **validation** NLL ({cal.get('fit_method', 'grid search')}); "
        f"validation NLL at T: {cal.get('val_nll_at_fitted_temperature', float('nan')):.4f}.\n"
        f"- Evaluated on the untouched **{split}** split; checkpoint `{cal.get('checkpoint', '?')}` "
        f"(sha256 `{str(cal.get('checkpoint_sha256', ''))[:12]}…`), fitted/evaluated {cal.get('evaluated_at', '?')}.\n"
        "- Temperature scaling never changes the predicted class (the arg-max), so accuracy is identical "
        "before and after; it only rescales the probabilities."
    )

    diagram = common.REPORTS_DIR / "reliability_diagram.png"
    if diagram.is_file():
        st.subheader("Reliability diagram")
        st.image(str(diagram), caption="Binned confidence vs accuracy, before and after calibration", width=640)
    else:
        st.info("reliability_diagram.png not found; rerun `run_evaluation --calibrate` to create it.")

    st.subheader("Download")
    path = common.REPORTS_DIR / "calibration.json"
    if path.is_file():
        common.download_bytes(path.read_bytes(), "calibration.json", "Download calibration JSON", "application/json")
    if diagram.is_file():
        common.download_bytes(diagram.read_bytes(), "reliability_diagram.png", "Download reliability diagram", "image/png")

    with st.expander("How to interpret this"):
        st.markdown(
            "- **ECE** (expected calibration error, 15 equal-width bins): average gap between confidence and "
            "accuracy; lower is better.\n"
            "- **NLL** (negative log-likelihood): penalises confident mistakes heavily; lower is better.\n"
            "- **Brier score**: mean squared error between probabilities and one-hot outcomes; lower is better.\n"
            "- On this well-calibrated model the numbers are already small, so temperature scaling changes "
            "them only slightly (possibly marginally worse ECE - reported as measured, not hidden).\n"
            "- Calibration was measured on this dataset only; probabilities on other labs/stains may be miscalibrated."
        )

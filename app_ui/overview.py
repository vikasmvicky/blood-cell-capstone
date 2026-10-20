"""Section A: Research Overview - project, dataset, model, verified results, limitations."""

from __future__ import annotations

import streamlit as st

from app_ui import common
from src.config import Config


def _render_project() -> None:
    st.title("\U0001f52c Explainable Blood-Cell Classification")
    st.markdown(
        "A research prototype that classifies **Peripheral Blood Cell (PBC) smear images** into eight cell "
        "types with a fine-tuned **MobileNetV2**, and explains each prediction with **Grad-CAM**. "
        "The project emphasises reproducibility: fixed data splits, checkpoint-verified loading, "
        "held-out evaluation, probability calibration and quantitative explanation-faithfulness tests."
    )
    st.warning(common.DISCLAIMER, icon="⚠️")


def _render_dataset(cfg: Config) -> None:
    st.subheader("Dataset")
    summary = common.load_report_json("split_summary.json")
    integrity = common.load_report_json("split_integrity.json")
    if summary is None:
        st.info("No split summary found (`outputs/reports/split_summary.json`).")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usable images", f"{summary['total_usable_images']:,}")
    c2.metric("Classes", len(summary["class_to_idx"]))
    c3.metric("Split seed", summary["random_seed"])
    c4.metric("Excluded (conflicts)", summary["total_excluded_images"])
    splits = summary["split_counts"]
    st.markdown(
        "Train / validation / test = "
        f"**{splits['train']:,} / {splits['val']:,} / {splits['test']:,}** "
        f"({', '.join(f'{k} {v:.1%}' for k, v in summary['split_percentages'].items())}). "
        "Exact-duplicate groups were kept inside one split to prevent leakage; two images with "
        "cross-class duplicate conflicts were excluded (see `excluded.csv`)."
    )
    if integrity is not None:
        kind = "ok" if integrity["status"] == "PASS" else "bad"
        st.markdown(
            common.status_chip(kind, f"Split integrity: {integrity['status']}")
            + " <span class='bca-note'>no exact-duplicate or duplicate-group leakage across "
            "train/val/test; labels and sampled files verified "
            "(`src/data/verify_splits.py`)</span>",
            unsafe_allow_html=True,
        )
    with st.expander("Per-class image counts per split"):
        cd = integrity.get("class_distribution") if integrity else None
        if cd:
            import pandas as pd

            st.dataframe(pd.DataFrame(cd), width="stretch")


def _render_model(cfg: Config) -> None:
    st.subheader("Model")
    bench = common.load_report_json("benchmark.json")
    predictor, err = common.get_predictor(cfg)
    if predictor is not None:
        loaded = predictor.loaded
        st.markdown(
            common.status_chip("ok", "Checkpoint loaded")
            + f" `{loaded.checkpoint_path.name}` · sha256 `{loaded.checkpoint_sha256[:16]}…` · "
            + (f"epoch {loaded.metadata.get('epoch')} · " if loaded.metadata.get("epoch") is not None else "")
            + f"{sum(p.numel() for p in loaded.model.parameters()):,} parameters · "
            + (f"validation-fitted temperature T={predictor.temperature:.3f}" if predictor.temperature else "no calibration applied"),
            unsafe_allow_html=True,
        )
    elif err is not None:
        common.render_checkpoint_error(cfg, err)
    rows = ["MobileNetV2 (ImageNet-pretrained, fine-tuned)", "Input 224×224, ImageNet normalisation"]
    if bench:
        rows.append(
            f"CPU latency batch-1: {bench['latency_ms_batch1']['median']:.1f} ms median "
            f"(p95 {bench['latency_ms_batch1']['p95']:.1f} ms) on this machine's CPU"
        )
    st.markdown("\n".join(f"- {r}" for r in rows))
    st.caption(
        "A from-scratch CNN baseline also exists in `src/models/cnn_model.py`, but its only local "
        "checkpoint was trained on a 20-sample smoke run, so it is **not** reported as a predictive baseline."
    )


def _render_verified_results() -> None:
    st.subheader("Current verified results")
    metrics = common.load_report_json("mobilenet_finetuned_test_metrics.json")
    if metrics is None:
        st.info("No evaluation report found. Run `python -m src.evaluation.run_evaluation --calibrate`.")
    else:
        st.markdown(
            common.status_chip("ok", "Measured")
            + f" <span class='bca-note'>held-out **{metrics.get('split', 'test')}** split "
            f"({metrics['test_samples']:,} images), checkpoint `{metrics.get('checkpoint', '?')}` "
            f"(`{str(metrics.get('checkpoint_sha256', ''))[:12]}…`), evaluated {metrics.get('evaluated_at', '?')}</span>",
            unsafe_allow_html=True,
        )
        c = st.columns(5)
        c[0].metric("Accuracy", common.pct(metrics["test_accuracy"]))
        c[1].metric("Macro F1", common.pct(metrics["macro_f1"]))
        c[2].metric("Weighted F1", common.pct(metrics["weighted_f1"]))
        c[3].metric("ECE (15 bins)", f"{metrics.get('ece_15_bins', float('nan')):.4f}")
        c[4].metric("NLL", f"{metrics.get('nll', float('nan')):.4f}")
    hist = common.load_report_json("historical_results.json")
    if hist is not None:
        st.markdown(
            common.status_chip("warn", "Historical")
            + f" <span class='bca-note'>the project's originally reported test accuracy of "
            f"**{100 * hist['test_accuracy']:.2f}%** (macro-F1 {100 * hist['macro_f1']:.2f}%) could **not** be "
            "reproduced with the available checkpoint; it is kept only as an unverified claim "
            "(`outputs/reports/historical_results.json`).</span>",
            unsafe_allow_html=True,
        )


def _render_experiment_status() -> None:
    st.subheader("Experiment status")
    entries = [
        ("mobilenet_finetuned_test_metrics.json", "Held-out test evaluation (accuracy, P/R/F1, confusion matrix)"),
        ("calibration.json", "Temperature scaling fitted on validation; before/after on test"),
        ("reliability_diagram.png", "Reliability diagram before/after calibration"),
        ("gradcam_faithfulness.json", "Grad-CAM deletion/insertion curves vs random baselines"),
        ("gradcam_faithfulness_curves.png", "Faithfulness curves plot"),
        ("robustness.json", "Accuracy under blur/noise/JPEG/brightness/contrast degradations"),
        ("segmentation_study.json", "Watershed centre-crop study (no retraining)"),
        ("benchmark.json", "Parameter count, size, CPU latency, throughput"),
        ("split_integrity.json", "Leakage/balance verification of the recorded split"),
        ("top_confusions.csv", "Most frequent class confusions"),
        ("test_predictions.csv", "Per-image test predictions for error analysis"),
    ]
    rows = []
    for name, desc in entries:
        kind, label = ("ok", "available") if common.report_exists(name) else ("warn", "not run")
        rows.append((common.status_chip(kind, label), desc, f"`{name}`"))
    st.markdown(
        "| Status | What it measures | Artifact |\\n|---|---|---|\\n"
        + "\\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows),
        unsafe_allow_html=True,
    )


def _render_downloads() -> None:
    st.subheader("Reports")
    for name, label, mime in [
        ("mobilenet_finetuned_test_metrics.json", "Test metrics (JSON)", "application/json"),
        ("mobilenet_finetuned_test_report.csv", "Per-class report (CSV)", "text/csv"),
        ("calibration.json", "Calibration (JSON)", "application/json"),
        ("gradcam_faithfulness.json", "Grad-CAM faithfulness (JSON)", "application/json"),
        ("robustness.json", "Robustness (JSON)", "application/json"),
        ("split_integrity.json", "Split integrity (JSON)", "application/json"),
    ]:
        path = common.REPORTS_DIR / name
        if path.is_file():
            common.download_bytes(path.read_bytes(), name, f"Download {label}", mime)


def _render_limitations() -> None:
    st.subheader("Limitations and intended use")
    st.markdown(
        "- **Not a medical device.** Educational/research prototype only; never use for diagnosis.\n"
        "- Trained and evaluated on **one dataset** (PBC, single acquisition/staining setup); other labs, "
        "microscopes or stains may behave very differently.\n"
        "- The model **always** picks one of the eight classes - including for images that are not blood "
        "cells, contain other cell types, or show several overlapping cells.\n"
        "- Softmax confidence is a model score. A validation-fitted temperature is applied only when "
        "`calibration.json` matches the loaded checkpoint exactly.\n"
        "- Grad-CAM heat-maps are coarse (7×7 grid upsampled) and qualitative; they do **not** prove the "
        "model reasons like a haematologist, and they are not segmentations.\n"
        "- The historical 97.07% accuracy was **not reproduced**; the measured value for this checkpoint "
        "is what the Evaluation Dashboard shows."
    )


def render(cfg: Config) -> None:
    _render_project()
    a, b = st.columns([3, 2], gap="large")
    with a:
        _render_dataset(cfg)
        _render_model(cfg)
        _render_verified_results()
    with b:
        _render_experiment_status()
        _render_downloads()
        _render_limitations()

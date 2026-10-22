# Objective Traceability Matrix

Source of truth: `app_ui/objectives.py` (pure Python, no Streamlit). This document is the
human-readable version; `tests/test_objectives.py` keeps the two consistent and honest.

The six objectives below are the ones defined in the capstone presentation. The deployed model
in this repository is a **fine-tuned MobileNetV2 whole-image classifier** (8 PBC classes) — not
an object detector and not a segmentation model. Statuses reflect what is actually implemented
and verifiable in this repository, not what a finished project would ideally contain.

## Status summary

| # | Objective | Status | Key evidence | Main gap |
|---|-----------|--------|--------------|----------|
| 1 | Reproduce and benchmark the YOLOv8 baseline | **Not implemented** | No YOLO code/dependency/checkpoint exists (verified by inspection) | No detection annotations in the PBC dataset; no `ultralytics` dependency |
| 2 | Upgrade to YOLOv9 / YOLO11 if supported | **Not implemented** | Same inspection as #1; only the classifier has trained weights | Blocked by #1 |
| 3 | Handle class imbalance (weighting / augmentation / oversampling) | **Implemented** | Inverse-frequency class-weighted CE in `src/models/train_mobilenet_finetune.py`; per-class test metrics in `outputs/reports/mobilenet_finetuned_test_metrics.json` | Sampler-variant run not archived in `outputs/reports/` |
| 4 | Improve WBC / platelet performance via resolution & config experiments | **Partial** | Per-class metrics + `segmentation_study.json` (watershed crop study, null result reported honestly) | No archived resolution sweep or alternative-architecture training runs |
| 5 | Web interface for upload and annotated results | **Implemented** | `app.py` + `app_ui/`; `tests/test_app_smoke.py` renders every page | "Annotations" are Grad-CAM overlays, not bounding boxes |
| 6 | Detection + downloadable summary report | **Partial** | Prediction JSON/CSV export; dashboard report downloads with correct MIME types | Detection/localisation not implemented (needs #1–#2) |

Overall: 2 implemented · 2 partial · 2 not implemented.

## The 2025 reference paper — scope boundary

Scientific Reports (2025), *"Efficient blood cell classification from microscopic smear images
using U-Net segmentation and a lightweight CNN"*, DOI: [10.1038/s41598-025-26947-5](https://doi.org/10.1038/s41598-025-26947-5).

This paper is a **methodological reference only**. Its pipeline (U-Net cell segmentation →
watershed separation → ROI extraction → lightweight CNN → **nine** categories including
erythrocyte) and its reported metrics belong to its own dataset and experimental setup. None of
its results are claimed, reproduced, or implied by this project. This repository classifies
8 PBC classes with MobileNetV2 and has **no** U-Net, no watershed-based ROI pipeline, no
erythrocyte class, and no segmentation ground truth to train or validate one.

---

## 1 · Reproduce and benchmark the existing YOLOv8 baseline — Not implemented

**Summary.** This repository contains no YOLOv8 code, no `ultralytics` dependency, no detector
checkpoint and no detection benchmark. The only trained model is a whole-image MobileNetV2
classifier (`outputs/models/best_mobilenet_finetuned.pt`). The PBC dataset used here provides
image-level class labels only — no bounding boxes or masks — so a YOLOv8 detection baseline
cannot be reproduced from the available data alone.

**Implemented in:** *(none)*

**Evidence:** *(none — the absence itself is the finding)*. Verified by repository inspection:
no YOLO modules under `src/`, no `ultralytics` in `requirements.txt`, no detection artifacts
under `outputs/`; `configs/config.json` declares architecture `mobilenet_v2`.

**Remaining gaps.**
- Add an object-detection framework (e.g. ultralytics) as a constrained dependency.
- Obtain or annotate detection ground truth (boxes/masks); the PBC images have class labels only.
- Train YOLOv8 on a fixed, leakage-safe split and benchmark against the classifier with clearly
  separated metrics (detection mAP vs classification accuracy).

**Limitations.** Marking this objective complete without detection ground truth would fabricate a
capability; it is recorded as not implemented instead.

## 2 · Upgrade to a newer YOLO architecture (YOLOv9 / YOLO11) — Not implemented

**Summary.** Blocked by objective 1: there is no YOLOv8 baseline here to upgrade from, and no
detection data to train any YOLO variant on. No YOLO dependency was added, because an untrained
detector would be an unusable showcase, not an upgrade.

**Implemented in:** *(none)* · **Evidence:** *(none)*. Same inspection as objective 1; the
delivered classifier is the only model with trained weights.

**Remaining gaps.** Complete objective 1 first; then compare YOLOv9/YOLO11 variants under
matched data, splits and compute, reporting both accuracy and latency.

**Limitations.** CPU-only development environment; modern YOLO training would need GPU
resources and annotated-data review.

## 3 · Handle class imbalance with weighting, augmentation and/or oversampling — Implemented

**Summary.** The delivered checkpoint was trained with **inverse-frequency class-weighted
cross-entropy** computed from the training-split counts
(`src/models/train_mobilenet_finetune.py`). A **WeightedRandomSampler** oversampling variant
exists as a documented experiment script (`src/models/train_mobilenet_sampler.py`). Training
adds mild augmentation (random flips, ≤15° rotations, `src/data/image_dataset.py`), the split
is stratified by duplicate group (`src/data/split_dataset.py`), and per-class
precision/recall/F1 are measured and displayed so imbalance effects stay visible.

**Evidence.**
- `outputs/reports/mobilenet_finetuned_training_history.csv` — training run of the weighted-CE
  configuration.
- `outputs/reports/mobilenet_finetuned_test_metrics.json` — per-class metrics on the
  2,562-image held-out test split: minority classes remain strong (basophil recall 0.967,
  platelet F1 0.993) while honest weaknesses stay exposed (basophil precision 0.843,
  IG recall 0.834).
- `outputs/reports/split_integrity.json` — split leakage/balance verification (PASS).

**Verification.** The Evaluation Dashboard renders these per-class numbers directly from the
saved report; `tests/test_objectives.py` pins the status and the evidence paths.

**Remaining gaps / limitations.** The sampler-variant run is not archived in
`outputs/reports/` (no `mobilenet_sampler_history.csv`), so the oversampling path is
implemented in code but has no recorded result in this archive. Residual per-class variance
shows weighting mitigates but does not eliminate imbalance.

## 4 · Improve WBC and platelet performance via resolution/model-config experiments — Partial

**Summary.** WBC and platelet performance is *measured* rigorously (per-class test metrics, top
confusions, robustness degradations, Grad-CAM faithfulness), and one controlled experiment
exists: a classical **watershed centre-crop study**
(`src/experiments/segmentation_study.py`) testing whether pre-cropping the cell region changes
classification accuracy — it did **not** improve accuracy, and the null result is reported as
measured. However, no input-resolution sweep or alternative model-configuration training runs
are archived; the deployed model is a single 224×224 MobileNetV2 configuration.

**Implemented in.** `src/experiments/segmentation_study.py`, `src/segmentation/watershed.py`,
`src/evaluation/robustness.py`.

**Evidence.** `outputs/reports/segmentation_study.json`, `outputs/reports/robustness.json`,
`outputs/reports/mobilenet_finetuned_test_metrics.json`, `outputs/reports/benchmark.json`.

**Remaining gaps.**
- Controlled resolution experiments (e.g. 160/224/320 px) with matched training budgets and
  validation-based selection.
- Alternative lightweight architectures trained and compared on the same split.
- Targeted error analysis of the IG ↔ basophil ↔ monocyte confusions to guide configuration
  changes.

**Limitations.** The reference paper's U-Net segmentation improvements cannot be claimed here:
this dataset ships no ground-truth masks, and the classical watershed experiment did not yield
an accuracy gain.

## 5 · Web interface for image upload and annotated results — Implemented

**Summary.** A Streamlit research application (`app.py` + `app_ui/`) supports PNG/JPG/JPEG
upload with validation (type, size, decodability, dimensions), prediction with top-k
probabilities and uncertainty indicators, Grad-CAM overlay as the annotation, heuristic
image-quality warnings, evaluation/calibration dashboards, and downloadable reports. It runs on
CPU and degrades gracefully when the checkpoint or reports are missing — it never substitutes
random weights.

**Implemented in.** `app.py`, `app_ui/` (prediction, explainability, evaluation, calibration,
overview, methodology), `src/inference/` (preprocess, predictor, quality),
`src/explainability/gradcam.py`.

**Evidence.** `tests/test_app_smoke.py` renders every section via Streamlit's AppTest harness
with valid, missing and corrupt checkpoints.

**Limitations.** "Annotated results" are Grad-CAM heat-map overlays, not bounding boxes; the app
explicitly states it is a research prototype, not a medical device.

## 6 · Detection and a downloadable summary report — Partial

**Summary.** The downloadable summary report is implemented: per-image prediction exports
(JSON/CSV with probabilities, uncertainty and warnings) plus dashboard downloads of the full
evaluation, calibration and robustness reports. Detection (localisation, bounding boxes, cell
counting) is **not** implemented — the model is a whole-image classifier, and drawing boxes or
counting cells would fabricate a capability the model does not have.

**Implemented in.** `app_ui/prediction.py`, `app_ui/evaluation_page.py`,
`app_ui/calibration_page.py`.

**Evidence.** `outputs/reports/mobilenet_finetuned_test_metrics.json`,
`outputs/reports/test_predictions.csv`, `outputs/reports/top_confusions.csv`.

**Remaining gaps.** Real detection/localisation depends on objectives 1–2 (annotated detection
data and a trained detector).

**Limitations.** No cell counting or localisation is offered; per-image summaries cover one
cell crop per upload, matching the training distribution.

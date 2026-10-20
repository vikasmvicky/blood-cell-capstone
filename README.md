# Explainable AI-Based Blood Cell Classification Using Deep Learning

Eight-class classification of Peripheral Blood Cell (PBC) smear images with a fine-tuned **MobileNetV2**,
a **Streamlit** website, **Grad-CAM** explanations, and a reproducible evaluation workflow.

> **Research and educational use only.** Not a medical device; output is not a clinical diagnosis.

Classes: basophil, eosinophil, erythroblast, immature granulocyte (IG), lymphocyte, monocyte, neutrophil, platelet.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # add requirements-dev.txt for pytest
# 1. put the trained checkpoint at outputs/models/best_mobilenet_finetuned.pt   (see ASSETS.md)
streamlit run app.py
```

The site runs on CPU. Without the checkpoint it still starts and tells you exactly which file is missing; it never
substitutes random or different weights.

## What the website does

Upload PNG/JPG/JPEG → predicted class + model confidence → ranked probabilities → Grad-CAM overlay → heuristic
image-quality warnings → limitations and disclaimer. Corrupt, empty, oversized, tiny, non-image or unsupported files get a
readable error. "Confidence" is the softmax score, **not** a calibrated probability or medical certainty (a temperature-scaled
value is shown only when `outputs/reports/calibration.json` was fitted for the exact checkpoint in use).

## Repository layout

```
app.py                         Streamlit website
configs/config.json            image size, normalisation, class names, paths, inference limits, seeds
src/config.py                  config loader (all paths relative to project root; env overrides)
src/data/                      inspect_dataset, split_dataset, image_dataset (existing; loader made path-portable)
src/models/                    cnn_model (baseline), train_*.py (existing), mobilenet.py (shared factory)
src/inference/                 preprocess, predictor (checkpoint validation), quality checks, service
src/explainability/gradcam.py  dependency-free Grad-CAM
src/evaluation/                run_evaluation, benchmark, robustness, metrics, calibration; evaluate_mobilenet.py (legacy)
src/segmentation/watershed.py  classical watershed cell separation (experimental, diagnostic)
src/experiments/               segmentation_study, gradcam_faithfulness
tests/                         pytest suite (synthetic fixtures; real-data tests skip when assets are absent)
outputs/reports/               small existing reports (split report, class map, histories)
docs/                          AUDIT.md, RESEARCH_NOTES.md, VERIFICATION.md
ASSETS.md                      excluded files and how to restore them
```

## Reproducible workflow

```bash
# Data (once; needs the PBC images, see ASSETS.md)
python -m src.data.inspect_dataset --data-dir data/raw/PBC_dataset_normal_DIB   # writes outputs/reports/image_manifest.csv
python -m src.data.split_dataset --seed 123                     # seed 123 = the recorded split (default is 42!)

# Train (optional; the existing checkpoint is normally used)
python -m src.models.train_mobilenet_finetune --epochs 5

# Evaluate on the held-out test split (reads test only; selects nothing on it)
python -m src.evaluation.run_evaluation --calibrate             # metrics, confusion matrix PNG, top confusions, calibration
python -m src.evaluation.benchmark                              # params, size, CPU latency, throughput
python -m src.evaluation.benchmark --architecture-only          # same, without the checkpoint (random weights, timing only)

# Research experiments (see docs/RESEARCH_NOTES.md; tune on --split val, report test once)
python -m src.evaluation.robustness --split test --max-images 500 [--ood-dir some/non_cell_images]
python -m src.experiments.segmentation_study --split val
python -m src.experiments.gradcam_faithfulness --split val

# Tests
pip install -r requirements-dev.txt && python -m pytest -q
```

## Results status (read this)

| Item | Status |
|---|---|
| MobileNetV2 test accuracy 97.07 %, macro-F1 97.20 % (`outputs/reports/historical_results.json`) | **Historical, not re-measured here** (checkpoint/test split not in the archive) |
| Parameter count of the 8-class head | Measured in this archive's tests/benchmark (see `docs/VERIFICATION.md`) |
| Segmentation / watershed improves accuracy | **Not claimed.** U-Net not implemented (no masks); watershed is an experiment with no result yet |
| Comparison with the Scientific Reports paper (97.10 %, nine classes) | Not like-for-like (different classes/splits); do not present as a head-to-head result |

See `docs/VERIFICATION.md` for exactly what was executed and what remains unverified.

## Known limitations

Single dataset and acquisition setup; the model always picks one of eight classes, even for non-cell images; softmax
confidence is uncalibrated unless calibrated; Grad-CAM is coarse (7×7) and qualitative; overlapping/multi-cell images are
outside the training distribution (PBC crops are one cell each).

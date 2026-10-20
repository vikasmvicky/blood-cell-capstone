# Verification report — what was actually run (2026-10-20)

Environment: **Windows 10** (10.0.19045), Intel Core i5-8350U (4C/8T, 1.70 GHz), **8 GB RAM, no GPU**,
Python 3.11.5, torch 2.14.1+cpu, torchvision 0.29.1+cpu, streamlit 1.66.0, scikit-learn 1.9.1,
pytest 9.1.1, 40 GB free disk. Assets available on this machine: full PBC dataset (17,092 images),
`splits.csv`, both checkpoints (`best_mobilenet_finetuned.pt` sha256 `d1f3c2b3…`, `best_cnn.pt`).

## Test suite

| Command | Result |
|---|---|
| `python -m pytest -q` | **73 passed, 0 failed, 0 skipped** (44 s) |

The suite covers: config/class mapping; dataset splits, shapes, label mapping, Windows-path fallback,
augmentation-only-on-train; inference preprocessing identical to the training path; valid/empty/garbage/
truncated/unsupported/tiny/huge uploads; checkpoint loading incl. missing/incompatible/corrupt and
class-order mismatch; prediction shapes, probability normalisation, temperature gating; Grad-CAM
shape/range/determinism and weight immutability; hand-computed metrics/confusion matrix; calibration
(ECE/NLL/Brier/temperature); reliability tables and plots; faithfulness helpers + CLI end-to-end on a
synthetic checkpoint; quality heuristics; robustness perturbation functions; watershed on synthetic
discs; evaluation and benchmark CLIs end-to-end on a synthetic dataset; and Streamlit AppTest runs of
**all six app sections** in three checkpoint states (missing/valid/corrupt).

**Important:** unit tests use *synthetic fixtures* (random weights, random pixels). They prove the code
paths work; model-quality claims come only from the real runs below.

## Real-asset runs (all exit code 0; logs in `outputs/logs/`)

| Command | Key result | Artifact |
|---|---|---|
| `python -m src.data.verify_splits` | **PASS** (11,964/2,564/2,562; no leakage) | `split_integrity.json` |
| `python -m src.evaluation.run_evaluation --calibrate` | acc **94.54 %**, macro-F1 **0.9413**, ECE 0.0106, NLL 0.1454, Brier 0.0782; T=0.903 | `mobilenet_finetuned_test_metrics.json`, `calibration.json`, `reliability_diagram.png`, `confusion_matrix.png`, `test_predictions.csv`, `top_confusions.csv` |
| `python -m src.evaluation.benchmark` | 2.234 M params; batch-1 median 37.1 ms; 25.6 img/s @32 | `benchmark.json` |
| `python -m src.experiments.gradcam_faithfulness --split val --max-images 60 --steps 20 --random-seeds 3` | deletion AUC 0.1904 vs random 0.2252; insertion 0.6771 vs 0.2266 | `gradcam_faithfulness.json`, `gradcam_faithfulness_curves.png` |
| `python -m src.evaluation.robustness --split test --max-images 200` | none 96.5 %; brightness×0.6 48.0 % (worst) | `robustness.json` |
| `python -m src.experiments.segmentation_study --split val --max-images 200` | crop 50.0 % vs baseline 96.5 % (**negative**) | `segmentation_study.json` |

## Application verification

| Check | Result |
|---|---|
| Streamlit AppTest — all 6 sections, **real** checkpoint + real reports | 0 exceptions, 0 errors |
| AppTest — real image upload (platelet crop) in Prediction Studio | predicted *Platelet*, 99.9 % (temperature-scaled), no exceptions |
| AppTest — same upload flow into Explainability | Grad-CAM rendered, no exceptions |
| AppTest — missing / corrupt checkpoint states | readable setup/error messages, no crash |
| Live server `streamlit run app.py --server.headless true --server.port 8505` | started with the real checkpoint; `/_stcore/health` **200**, `/` **200**; terminated afterwards |
| `python -m compileall app.py app_ui src` | OK |

## NOT verified / limitations of this verification

- **Browser-level visual rendering** (CSS layout, slider interaction, download clicks): no browser
  automation was available. Verified via Streamlit's AppTest (element-level DOM simulation, including a
  real image upload) plus live-server HTTP checks — not via screenshots.
- GPU execution, macOS/Linux execution, Python versions other than 3.11.
- The full Scientific Reports reference paper text (bot-blocked earlier); its 97.10 % figure is not quoted
  as a comparison anywhere.
- Retraining to reproduce the historical 97.07 % (not attempted; see `LIMITATIONS.md`).
- Behaviour on any dataset other than PBC.

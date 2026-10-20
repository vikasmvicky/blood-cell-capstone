# Research improvement report — measured results

All numbers below were **measured on 2026-10-20** on this machine with the real checkpoint
(`best_mobilenet_finetuned.pt`, sha256 `d1f3c2b3…`) and the real recorded split (seed 123), unless a row
says otherwise. Machine-readable sources live in `outputs/reports/`; console logs in `outputs/logs/`.

## 1 · Held-out test evaluation (measured, reproduced)

Command: `python -m src.evaluation.run_evaluation --calibrate` — 2,562 test images, CPU, batch 32.

| Metric | Value |
|---|---|
| Accuracy | **94.54 %** (0.94536) |
| Macro precision / recall / F1 | 0.9386 / 0.9463 / **0.9413** |
| Weighted precision / recall / F1 | 0.9463 / 0.9454 / 0.9450 |
| NLL / ECE(15) / Brier | 0.1454 / 0.0106 / 0.0782 |

Per-class F1 (support): eosinophil 0.998 (467) · platelet 0.993 (352) · neutrophil 0.953 (499) ·
erythroblast 0.960 (232) · lymphocyte 0.948 (182) · monocyte 0.912 (213) · basophil 0.901 (183) ·
**IG 0.865 (434)** — the weakest class.

Top confusions: IG→basophil 31 (7.1 % of IG), IG→neutrophil 24 (5.5 %), monocyte→IG 19 (8.9 %),
neutrophil→IG 16 (3.2 %) — the granulocyte family plus monocytes dominates all errors, which is the
expected clinically hard neighbourhood.

**The historical 97.07 % claim was NOT reproduced** (measured 94.54 %, a 2.5-point gap). It is retained
only as a labelled historical claim in `outputs/reports/historical_results.json`. Possible causes
(documented, not proven): different checkpoint than the one originally reported, different evaluation
code path, or an unrecorded data/augmentation state. The measured number is the only one presented in
the app and README.

## 2 · Calibration (measured)

Temperature T = **0.903** fitted by validation-NLL grid search (val NLL 0.1409), then evaluated on test:

| | Before | After |
|---|---|---|
| NLL | 0.1454 | **0.1437** ↓ |
| ECE (15 bins) | 0.0106 | 0.0110 (marginally worse) |
| Brier | 0.0782 | 0.0786 (marginally worse) |
| Accuracy | 0.94536 | 0.94536 (unchanged by construction) |

Interpretation: softmax outputs on this model are already well calibrated (ECE ≈ 1 %), so scaling has
almost nothing to fix; NLL improves slightly while ECE/Brier shift within noise. Reported honestly,
including the marginal worsening. The app applies T only when `calibration.json` matches the loaded
checkpoint's sha256.

## 3 · Grad-CAM faithfulness (measured)

Command: `python -m src.experiments.gradcam_faithfulness --split val --max-images 60 --steps 20 --random-seeds 3`
(deletion + insertion curves, mean-colour baseline, 21 fractions, 3 random-order baselines per image).

| AUC | Grad-CAM | Random | Faithful direction? |
|---|---|---|---|
| Deletion (lower better) | **0.1904** | 0.2252 | yes |
| Insertion (higher better) | **0.6771** | 0.2266 | yes |

Both curves separate clearly from the random baseline over 60 images — evidence (not proof) that the
heat-maps highlight evidence the model actually uses. Plot: `outputs/reports/gradcam_faithfulness_curves.png`.

## 4 · Robustness under degradations (measured, 200 test images, seed 42)

| Perturbation | Accuracy |
|---|---|
| none | 96.5 % |
| Gaussian blur r=1 | 94.5 % |
| Gaussian blur r=3 | 63.0 % |
| Noise σ=10 | 90.0 % |
| Noise σ=25 | 77.5 % |
| JPEG q=30 | 92.5 % |
| JPEG q=10 | 78.5 % |
| Brightness ×0.6 | **48.0 %** |
| Brightness ×1.5 | 53.0 % |
| Contrast ×0.5 | 74.0 % |
| Downscale ×0.25 + up | 81.0 % |

The model is fairly tolerant to blur/noise/compression but **highly sensitive to brightness and contrast
shifts** — an important limitation for use on slides from other staining protocols.

## 5 · Watershed centre-crop study (measured, 200 val images) — negative result

Baseline (whole image) accuracy 96.5 % vs watershed centre-crop 50.0 % (Δ = −46.5 points). The watershed
over-segments PBC crops (mean 7-13 "objects"/image; the textured background is not separable), so cropping
destroys context. **Conclusion: do not use the crop pipeline; no segmentation claim is made.**

## 6 · Benchmark (measured, this hardware)

2,234,120 parameters · 8.94 MB fp32 weights · checkpoint file 9.18 MB · batch-1 CPU latency
median 37.1 ms (mean 38.4, p95 49.3) · throughput 25.6 img/s at batch 32 · torch 2.14.1+cpu, 4 threads,
Intel i5-8350U, Windows 10 (`outputs/reports/benchmark.json`). Hardware-specific; rerun elsewhere to
quote different numbers.

## 7 · Split integrity (measured)

`python -m src.data.verify_splits` → PASS: counts 11,964/2,564/2,562 match `split_summary.json`; zero
sha256 or duplicate-group leakage across splits; labels consistent; 500-file sample all present
(`outputs/reports/split_integrity.json`).

## 8 · Improvements implemented vs not done

| Improvement | Status |
|---|---|
| Six-section research application (overview, prediction, explainability, evaluation, calibration, methodology) | Implemented (`app.py`, `app_ui/`) |
| Strict checkpoint loading, sha256 provenance, no random-weight fallback | Implemented + tested |
| Reproducible evaluation CLI with per-class metrics, confusion matrices, error analysis | Implemented + run on real data |
| Calibration (ECE/NLL/Brier, val-fitted temperature, reliability diagrams) | Implemented + run on real data |
| Grad-CAM faithfulness (deletion+insertion, AUC, multi-seed random baselines, plots) | Implemented + run on real data |
| Robustness/degradation study | Implemented + run on real data |
| Split-integrity verification CLI | Implemented + run (PASS) |
| Benchmark (latency/throughput/size) | Implemented + run on real data |
| Prediction export (CSV/JSON) | Implemented |
| U-Net segmentation | **Not done — no ground-truth masks exist; would be fabricated science** |
| CNN-vs-MobileNet comparison | **Not done — CNN checkpoint is a smoke run, not a trained baseline; comparison would be dishonest** |
| Retraining to chase 97.07 % | **Not done — no evidence the historical figure is reachable with this checkpoint/protocol; would require GPU time** |

# Repository audit — `D:\blood-cell-capstone` (branch `research-completion`)

Audited 2026-10-20 on Windows 10 (i5-8350U, 8 GB RAM, no GPU) with Python 3.11.5,
torch 2.14.1+cpu, streamlit 1.66, pytest 9.1.1.

## What existed before this work (original `main`)

| Area | State found |
|---|---|
| `app/`, `configs/`, `docs/`, `notebooks/` | **Empty directories** — no application, no config, no docs |
| `src/data/` | `inspect_dataset.py` (scan/hash/duplicates), `split_dataset.py` (leakage-aware 70/15/15 split, seed 123 recorded), `image_dataset.py` (PyTorch dataset) |
| `src/models/` | `cnn_model.py` (from-scratch CNN), `train_cnn.py`, `train_mobilenet.py`, `train_mobilenet_sampler.py`, `train_mobilenet_finetune.py` (produced the final checkpoint) |
| `src/evaluation/evaluate_mobilenet.py` | Legacy test-split evaluation (overall metrics only) |
| `src/inference/`, `src/training/` | Empty packages |
| `tests/` | `test_split_dataset.py` (real pytest tests) + three scripts that ran code at import time or collected no tests |
| `outputs/models/` | `best_mobilenet_finetuned.pt` (9.18 MB, sha256 `d1f3c2b3…`), `best_cnn.pt` (0.42 MB) |
| `outputs/reports/` | class map, split summary/report, splits.csv (17,090 rows), dataset summary, exact duplicates, training histories, a 558-byte legacy `mobilenet_finetuned_test_metrics.json` |
| `data/raw/PBC_dataset_normal_DIB/` | 17,092 images in 8 class folders (present on this machine) |
| `requirements.txt` | Contained **unresolved git merge-conflict markers** (`<<<<<<< HEAD`, `>>>>>>> origin/main`) and a UTF-8 BOM |
| `README.md` | Contained **unresolved merge-conflict markers** between two README versions |
| Root | 281 MB `blood_cell_colab_package.zip` + seven 40 MB `blood_cell_part_*.bin` split chunks |

## Findings and resolutions

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | Merge-conflict markers committed in `requirements.txt` and `README.md` — neither file was installable/readable as-is | Blocker | Both rewritten cleanly (conflict-free, BOM removed) |
| 2 | No application of any kind existed in the primary repo (`app/` empty) | Blocker | Full six-section Streamlit research application built (`app.py` + `app_ui/`) |
| 3 | No model-loading/inference code; training scripts only | High | `src/inference/` package: strict checkpoint validation (sha256, class order, architecture), shared preprocessing identical to training, quality heuristics |
| 4 | No reproducible evaluation entry point; legacy evaluator reported overall metrics only | High | `src/evaluation/run_evaluation.py`: per-class metrics, confusion matrices, top confusions, per-image predictions, NLL/ECE/Brier, optional temperature scaling |
| 5 | Manifest stored absolute `D:\…` paths, breaking portability | Medium | `resolve_image_path` falls back to `data_root/rel_path` then `data_root/<class>/<file>`; tested |
| 6 | `split_dataset.py` default seed 42 vs recorded split seed 123 | Medium | Documented in README/ASSETS; `--seed 123` reproduces the recorded split; verification CLI added (`src/data/verify_splits.py`) |
| 7 | No explanation method despite the project title | High | Grad-CAM (`src/explainability/gradcam.py`) + quantitative faithfulness experiment (deletion/insertion curves vs random baselines, multi-seed, AUC, plots) |
| 8 | No calibration analysis | Medium | ECE/NLL/Brier, validation-fitted temperature scaling, reliability diagrams (`src/evaluation/calibration.py`) |
| 9 | `mobilenet_finetuned_training_history.csv` contains only 1 epoch while scripts cite best val macro-F1 0.9754 | Medium | Flagged as historical/unverified; not edited |
| 10 | `cnn_training_history.csv` is a 20-sample smoke run (acc 0.15), not a baseline result | Medium | CNN **not** reported as a baseline; documented |
| 11 | Historical 97.07 % test accuracy claim | Medium | Re-measured with the real checkpoint: **94.54 %**. 97.07 % kept only as labelled historical claim (`historical_results.json`) |
| 12 | `test_image_loader_manual.py` ran at import time; other test files collected no tests | Medium | All wrapped into proper pytest tests; suite now 73 tests |
| 13 | No segmentation ground truth in the PBC dataset | Info | No segmentation accuracy claimed; watershed kept only as a classical utility behind a measurement study (which measured a large *negative* effect of cropping) |
| 14 | Large artifacts in repo root (281 MB zip + 40 MB chunks) | Medium | `.gitignore` extended; excluded from the deliverable ZIP |
| 15 | Deprecated `use_container_width` API (Streamlit removal after 2025-12-31) | Low | Migrated to `width="stretch"` |

## Cross-checks performed

- Class mapping `basophil…platelet → 0…7` identical in `class_to_idx.json`, checkpoint metadata and config; loader fails on any mismatch (tested).
- `splits.csv` (17,090 rows) re-verified against images on disk: counts, ratios, per-class distribution, label consistency, sha256 and duplicate-group leakage all **PASS** (`outputs/reports/split_integrity.json`).
- Both checkpoints byte-identical between the primary repo and the reference working copy (`sha256 d1f3c2b3…` / `abd328f8…`).
- The two excluded images (`EO_225902.jpg`, `BNE_191112.jpg`) are the documented cross-class duplicate conflict (identical bytes, two labels); they remain listed in `excluded.csv`, not deleted from disk.
- Inference preprocessing asserted equal to the training dataset path by `tests/test_preprocess.py`.

# Reproducibility

## Pinned environment (verified)

Python 3.11.5 · torch 2.14.1+cpu · torchvision 0.29.1+cpu · streamlit 1.66.0 · scikit-learn 1.9.1 ·
scikit-image 0.26.0 · numpy 2.4.6 · pandas 3.0.6 · matplotlib 3.11.2 · Pillow 12.3.0 · pytest 9.1.1.
`requirements.txt` uses lower bounds (`>=`) for portability; `requirements-dev.txt` adds pytest.

## Determinism levers

| Lever | Where |
|---|---|
| Split seed **123** (recorded split) | `src/data/split_dataset.py --seed 123`; recorded in `split_summary.json` |
| Experiment/app seed **42** | `configs/config.json` → `reproducibility.seed`; `src/utils/seed.py` sets `random`/`numpy`/`torch` |
| Faithfulness sampling seed | `gradcam_faithfulness --seed` (default: config seed); random baselines use derived seeds 1000+i |
| Checkpoint identity | sha256 recorded in every report; loader refuses mismatched class order/architecture |
| Config single-source | `configs/config.json` (image size, normalisation, paths, thresholds) |
| Temperature gating | app applies T only if `calibration.json` sha256 matches the loaded checkpoint |

CPU-only torch is deterministic for these workloads (no cuDNN autotune). Exact floating-point equality
across different CPU architectures is not guaranteed; metric differences beyond the 4th decimal should be
treated as platform noise.

## Full reproduction sequence (Windows PowerShell)

```powershell
cd D:\blood-cell-capstone
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt

# Data (only if outputs/reports/splits.csv or images are missing)
python -m src.data.inspect_dataset --data-dir data/raw/PBC_dataset_normal_DIB
python -m src.data.split_dataset --seed 123
python -m src.data.verify_splits

# Evaluation, calibration, benchmark (needs outputs/models/best_mobilenet_finetuned.pt)
python -m src.evaluation.run_evaluation --calibrate
python -m src.evaluation.benchmark

# Experiments
python -m src.experiments.gradcam_faithfulness --split val --max-images 60 --steps 20 --random-seeds 3
python -m src.evaluation.robustness --split test --max-images 200
python -m src.experiments.segmentation_study --split val --max-images 200   # measured negative result

# App
streamlit run app.py

# Tests
python -m pytest -q
```

## Artifact provenance

Every report under `outputs/reports/` records (where applicable): checkpoint name + sha256, split name,
sample counts, seed, and timestamp. `outputs/logs/*.log` contain the exact console output and exit codes
of the 2026-10-20 runs. Results marked "historical" (`historical_results.json`) come from the original
project author and were never re-measured.

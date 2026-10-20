# Repository audit (uploaded `blood-cell-capstone-main.zip`)

## What existed (42 entries, ~75 KB uncompressed)

- `src/data/`: `inspect_dataset.py` (scan, hash, duplicates), `split_dataset.py` (leakage-aware 70/15/15 split, tested), `image_dataset.py` (PyTorch dataset).
- `src/models/`: `cnn_model.py` (from-scratch CNN baseline), `train_cnn.py`, `train_mobilenet.py`, `train_mobilenet_sampler.py`, `train_mobilenet_finetune.py` (the run that produced the final checkpoint).
- `src/evaluation/evaluate_mobilenet.py`: test-split evaluation (legacy; still present and unchanged).
- `tests/`: `test_split_dataset.py` (pytest), plus three scripts that needed real data or lacked `test_` functions.
- `outputs/reports/`: class map, split report/summary, dataset summary, duplicates, two training-history CSVs.
- Empty packages `src/inference/` and `src/training/`; `package_for_colab.py`, `split_package.py`; `.gitignore`, `README.md`, `requirements.txt`.

## Missing assets (not in the ZIP, not fabricated)

Trained checkpoint (`outputs/models/best_mobilenet_finetuned.pt`), PBC images, `splits.csv`, `image_manifest.csv`, the training notebook, MobileNet test-metrics files, `mobilenet_finetuned_summary.json`.

## Findings and what was done

| Finding | Action |
|---|---|
| `README.md` contained unresolved git merge markers (`<<<<<<< HEAD`); README, `.gitignore`, `requirements.txt` had a UTF-8 BOM | Rewritten cleanly |
| `requirements.txt` lacked torch, torchvision, tqdm, scipy, scikit-image, pytest; listed unused jupyter/seaborn | Fixed; dev requirements split out |
| `src/inference/` empty; no app, no config | Added inference package, `app.py`, `configs/config.json` |
| Manifest stores absolute Windows paths (`D:\...`) so the dataset loader cannot run elsewhere | `resolve_image_path` falls back to `data_root/rel_path` and `data_root/class/filename`; covered by a test |
| Training uses [0,1] tensors then ImageNet normalisation in the trainer; eval repeats it by hand; nothing for single images | One shared inference preprocess; a test asserts it equals the dataset+normalisation path |
| `split_dataset.py` default seed is 42 but the recorded split (`split_summary.json`) used 123 | Documented; README/ASSETS say `--seed 123` |
| `train_mobilenet_finetune.py` printed a hard-coded `/5` | Now `args.epochs` (no other behaviour change) |
| `mobilenet_finetuned_training_history.csv` has one epoch (val macro-F1 0.9455) while the script header cites best val macro-F1 0.9754 | Flagged: the CSV is not the full history of the run that produced the checkpoint. Not edited |
| `cnn_training_history.csv` is a 20-sample smoke test (acc 0.15) | Flagged: not a CNN baseline result |
| `test_image_loader_manual.py` ran code at import time (breaks `pytest` collection); `test_augmentation.py`/`test_cnn_model.py` had no collected tests | Wrapped; real-data tests skip when assets are absent |
| Three MobileNet trainers (frozen / sampler / fine-tune) | Kept; only fine-tune is tied to the reported checkpoint |
| `evaluate_mobilenet.py` duplicates what `run_evaluation.py` now does | Legacy file kept unchanged; `run_evaluation.py` is the documented path (same output filenames and metric keys) |
| `exact_duplicates.json` / `excluded.csv` embed `D:\` paths | Left as historical reports (not read by code) |

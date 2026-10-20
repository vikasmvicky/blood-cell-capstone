# Verification report (what was actually run)

Environment: Linux sandbox, Python 3.12.3, **1 CPU core**, no GPU, torch 2.14.1, torchvision 0.29.1, streamlit 1.66.0,
scikit-learn 1.8.0, scikit-image 0.26.0, pytest 9.1.1. No dataset and no trained checkpoint were available.

## Executed and passed

| Check | Result |
|---|---|
| `python -m pytest -q` | **57 passed, 2 skipped** (the 2 skips are the original real-data tests: `splits.csv` / images absent) |
| `python -m compileall src app.py tests` | OK |
| `pyflakes` on new code | no errors (unused imports only remain in original files) |
| Streamlit server (`streamlit run app.py --server.headless true`) | started; `/_stcore/health` and `/` returned HTTP 200 (run without a checkpoint) |
| Streamlit AppTest: no checkpoint / valid (synthetic) checkpoint / corrupt checkpoint | all three render without exception and show the expected setup message, waiting-for-upload state, or load error |
| MobileNetV2 8-class head parameter count | **2,234,120** (= 2.23 M, matches the historical figure) |
| `python -m src.evaluation.benchmark --architecture-only` (random weights; timing/size only) | 8.94 MB fp32 weights; batch-1 CPU latency mean 17.72 ms, median 17.72 ms, p95 19.05 ms; throughput 21.5 img/s (batch 32); **1 thread on a 1-core VM** - hardware-specific, saved as `outputs/reports/benchmark_architecture_only_sandbox.json` |

What the tests cover: config/class mapping; dataset splits, shapes, label mapping, Windows-path fallback, augmentation only on train;
inference preprocessing equal to the training path; valid/empty/garbage/truncated/unsupported/tiny/huge/oversized uploads;
checkpoint loading, missing/incompatible/corrupt checkpoints, class-order mismatch; prediction shape, ranked probabilities;
Grad-CAM shape/range/determinism and that it does not modify weights; hand-computed metrics and confusion matrix; calibration;
quality heuristics; perturbations; watershed on synthetic discs (overlapping discs split into 2); evaluation and benchmark CLIs
end-to-end on a synthetic dataset.

**Important:** the model-related tests use a *random-weight synthetic checkpoint* and *synthetic images*. They prove the code
path works, not that the model is accurate.

## NOT verified (assets or capability unavailable)

- Any accuracy/F1 number: the 97.07 % result is historical and was **not reproduced**. Run `python -m src.evaluation.run_evaluation`.
- Loading the author's real `best_mobilenet_finetuned.pt` (format inferred from the trainer code; the loader validates it strictly).
- Behaviour on real PBC images: Grad-CAM quality, quality-warning thresholds, watershed parameters, robustness numbers, OOD AUROC, calibration temperature.
- `segmentation_study`, `gradcam_faithfulness`, `robustness` CLIs on real data (only their shared plumbing and helper functions are tested).
- Real-data tests `test_augmentation_on_real_splits`, `test_real_split_sizes_and_batches` (skipped).
- Training scripts and `split_dataset`/`inspect_dataset` CLIs on the full dataset (unchanged code, not run).
- Browser file-upload flow end to end (Streamlit AppTest cannot upload files); the same logic is covered via `predict_upload` tests.
- GPU execution, Windows/macOS execution, and Python versions other than 3.12.
- The full text of the reference paper (blocked); its 97.10 % figure and Grad-CAM content.

# Methodology

Everything below is what the code in this repository **actually does**; commands reproduce each step.
Reported numbers are measured on this machine (see `VERIFICATION.md`) unless labelled historical.

## 1 · Dataset

- **Source**: Peripheral Blood Cell (PBC) dataset, `data/raw/PBC_dataset_normal_DIB`, 17,092 JPEG images,
  8 classes: basophil, eosinophil, erythroblast, immature granulocyte (`ig`), lymphocyte, monocyte,
  neutrophil, platelet.
- **Scan**: `python -m src.data.inspect_dataset --data-dir data/raw/PBC_dataset_normal_DIB` verifies every
  image (decodability, dimensions, format) and computes a SHA-256 per file →
  `outputs/reports/image_manifest.csv` (17,092 rows, 0 unreadable).
- **Class counts**: basophil 1,218 · eosinophil 3,117 · erythroblast 1,551 · IG 2,895 · lymphocyte 1,214 ·
  monocyte 1,420 · neutrophil 3,329 · platelet 2,348.

## 2 · Duplicates, exclusions and splitting

- Exact duplicates are grouped by content hash. **Two images** (`EO_225902.jpg`, `BNE_191112.jpg`) carry
  identical bytes under *two different class labels* — an unresolvable label conflict — and are excluded
  from splitting (listed in `outputs/reports/excluded.csv`; the files themselves are left on disk).
- `python -m src.data.split_dataset --seed 123` assigns whole duplicate **groups** to splits
  (70/15/15, stratified), so identical or grouped near-identical crops cannot leak across
  train/validation/test. Recorded split: **11,964 / 2,564 / 2,562** images
  (70.01/15.00/14.99 %) → `outputs/reports/splits.csv` (machine-readable manifest).
- `python -m src.data.verify_splits` independently re-checks counts, ratios, per-class distributions,
  label consistency, file presence and sha256/duplicate-group leakage →
  `outputs/reports/split_integrity.json` (**PASS** on 2026-10-20).
- The **validation** split is used for any fitting (temperature). The **test** split is touched only by
  final one-shot measurements.

## 3 · Preprocessing and augmentation

- Every image: RGB → PIL bilinear resize to 224×224 (no aspect preservation, matching the training
  pipeline) → ÷255 → ImageNet mean/std normalisation.
- Training only: random horizontal/vertical flips (p=0.5 each) and rotation ≤15°
  (`BloodCellDataset.augment_image`).
- The inference path (`src/inference/preprocess.py`) is asserted **tensor-identical** to the
  dataset path by `tests/test_preprocess.py`.

## 4 · Models and training

- **MobileNetV2** (torchvision, ImageNet-pretrained) with `classifier[1]` replaced by
  `Linear(1280, 8)` — 2,234,120 parameters. Fine-tuning script:
  `python -m src.models.train_mobilenet_finetune --epochs 5`. The shipped checkpoint is
  `outputs/models/best_mobilenet_finetuned.pt` (sha256 `d1f3c2b3…`).
- **From-scratch CNN baseline** (`src/models/cnn_model.py`) exists, but its only local checkpoint was a
  20-sample smoke run (val acc 0.15); it is therefore **not** reported as a predictive baseline.
  No architecture comparison is claimed because only one genuinely trained model exists locally.

## 5 · Evaluation methodology

`python -m src.evaluation.run_evaluation --calibrate`

- Loads the checkpoint strictly (`weights_only=True`, architecture, class order vs `class_to_idx.json`,
  sha256 recorded); never falls back to random weights.
- Runs the held-out **test** split once (batch inference), computes: accuracy, macro/weighted
  precision/recall/F1, per-class P/R/F1/support, raw + row-normalised confusion matrices, top confusions,
  per-image predictions (`test_predictions.csv`, machine-path-free), NLL, ECE (15 equal-width bins),
  multiclass Brier score.
- With `--calibrate`: fits a single temperature by deterministic grid search minimising **validation**
  NLL, then reports before/after NLL/ECE/Brier on test (accuracy unchanged by construction) and writes
  `calibration.json` + `reliability_diagram.png`.
- Benchmark: `python -m src.evaluation.benchmark` — parameter count, weight size, batch-1 CPU latency
  (mean/median/p95 over 50 timed calls after warm-up), batched throughput, hardware/software recorded.

## 6 · Explainability methodology

- Grad-CAM (Selvaraju et al., ICCV 2017) on the last convolutional block (`model.features[-1]`, 7×7
  grid): class-score gradients are global-average-pooled into per-channel weights; the weighted
  activation sum passes ReLU and is bilinearly upsampled to 224×224 and normalised to [0, 1]
  (`src/explainability/gradcam.py`). Implementation is dependency-free (torch + numpy + matplotlib).
- Overlays blend a jet heat-map over the model input at a user-controlled opacity.
- The map is coarse and qualitative; a plausible-looking map is **not** evidence of faithful reasoning —
  hence the quantitative experiment below.

## 7 · Calibration methodology

`src/evaluation/calibration.py`: softmax(ECE, NLL, Brier) helpers; temperature scaling
(logit division by a scalar T) fitted on validation only. Measured on this checkpoint: T = 0.903,
test NLL 0.1454 → 0.1437, ECE 0.0106 → 0.0110, Brier 0.0782 → 0.0786, accuracy unchanged
(`outputs/reports/calibration.json`). The model is already well calibrated, so the effect is small —
reported as measured, including the marginal ECE/Brier worsening.

## 8 · Faithfulness experiments

`python -m src.experiments.gradcam_faithfulness --split val --max-images 60 --steps 20 --random-seeds 3`

- Per image: Grad-CAM of the **predicted** class (fixed for the whole curve).
- **Deletion curve**: pixels replaced one order-position at a time by the per-image mean colour,
  highest saliency first; target-class probability tracked at 21 fractions (0-100 %).
- **Insertion curve**: from the mean-colour blank, original pixels inserted in saliency order.
- **Random baseline**: same procedure with random pixel orders, averaged over 3 seeds.
- Curves summarised by trapezoidal AUC; mean ± std across 60 validation images; plots +
  machine-readable JSON. Measured: deletion AUC 0.1904 (Grad-CAM) vs 0.2252 (random); insertion AUC
  0.6771 vs 0.2266 — both directions favour Grad-CAM, supporting (not proving) faithfulness.

## 9 · Robustness

`python -m src.evaluation.robustness --split test --max-images 200` measures accuracy under blur, Gaussian
noise, JPEG re-compression, brightness/contrast shifts and downscaling (perturbations applied to the
original image before normal preprocessing). Optional `--ood-dir` AUROC probe of max-softmax as an
out-of-distribution score. Results are dataset-specific; no cross-lab behaviour is implied.

## 10 · Segmentation — explicitly out of scope

The PBC dataset provides **no ground-truth segmentation masks**. Training a segmentation network on
classical pseudo-masks would only imitate that classical method and prove nothing, so no segmentation
model or accuracy claim exists. `src/segmentation/watershed.py` is a classical utility used only by
`src/experiments/segmentation_study.py`, which *measured* that centre-cell cropping with the existing
model changes accuracy from 96.5 % to 50.0 % on 200 validation images (a large **negative** result —
the watershed over-segments the textured background, 7-13 "objects" per image).

## 11 · Reproducibility

Fixed seeds (split 123; app/experiments 42 unless stated), single config file `configs/config.json`,
strict checkpoint validation with sha256 in every report, machine-readable outputs, and a pytest suite
(73 tests) using synthetic fixtures to cover all code paths without real assets. See `REPRODUCIBILITY.md`.

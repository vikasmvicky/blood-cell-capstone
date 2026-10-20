"""Section F: Research and Methodology - dataset, training, evaluation and experiment protocols."""

from __future__ import annotations

import streamlit as st

from app_ui import common
from src.config import Config


def render(cfg: Config) -> None:
    st.title("Research and Methodology")
    st.caption("How the data, splits, model, evaluation and explanations were produced - and how to reproduce them.")

    integrity = common.load_report_json("split_integrity.json")
    bench = common.load_report_json("benchmark.json")
    faith = common.load_report_json("gradcam_faithfulness.json")

    st.markdown(
        """
### 1 · Dataset preparation
The Peripheral Blood Cell (PBC) dataset (`data/raw/PBC_dataset_normal_DIB`) contains 17,092 images in 8 classes
(basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte, neutrophil, platelet).
`src/data/inspect_dataset.py` scanned every image (readability, size, SHA-256) into
`outputs/reports/image_manifest.csv`. Two images were excluded because identical bytes appeared under two
different class labels (cross-class conflicts); they are listed in `outputs/reports/excluded.csv` and were
**not** silently deleted.

### 2 · Duplicates and splitting
Exact duplicates were grouped by content hash. `src/data/split_dataset.py` assigns whole duplicate *groups*
(not individual images) to splits, so identical/near-identical crops can never leak between train, validation
and test. The recorded split uses **seed 123**, ratios 70/15/15, and is stored machine-readably in
`outputs/reports/splits.csv`. `src/data/verify_splits.py` re-checks counts, label consistency, file presence
and cross-split leakage independently and writes `outputs/reports/split_integrity.json`.

### 3 · Preprocessing and augmentation
All images: RGB → bilinear resize to 224×224 → ÷255 → ImageNet mean/std. Training only adds mild random
flips and ≤15° rotations (`BloodCellDataset.augment_image`). The inference path
(`src/inference/preprocess.py`) reproduces the training tensors exactly; a test asserts the two paths agree.

### 4 · Model and training
MobileNetV2 (ImageNet-pretrained) with the classifier head replaced by a linear 8-class layer
(2.23 M parameters). Trained with the fine-tuning script `src/models/train_mobilenet_finetune.py`; the
resulting checkpoint is `outputs/models/best_mobilenet_finetuned.pt`. A from-scratch CNN baseline exists in
`src/models/cnn_model.py` but was only smoke-trained locally, so it is **not** reported as a baseline.

### 5 · Evaluation methodology
`python -m src.evaluation.run_evaluation` loads the checkpoint strictly (sha256 recorded), runs the held-out
test split once, and writes accuracy, macro/weighted precision/recall/F1, per-class metrics, confusion
matrices, top confusions, per-image predictions, NLL/ECE/Brier. No hyper-parameter is selected on the test
set. Benchmark: `python -m src.evaluation.benchmark` (parameter count, file size, batch-1 CPU latency
mean/median/p95, batched throughput) with hardware recorded in `outputs/reports/benchmark.json`."""
    )
    if bench:
        st.caption(
            f"Measured on this machine: {bench.get('latency_ms_batch1', {}).get('median', '?')} ms median "
            f"batch-1 latency, {bench.get('throughput_images_per_s', '?')} img/s at batch "
            f"{bench.get('throughput_batch_size', '?')} ({bench.get('cpu', '?')}, torch {bench.get('torch_version', '?')})."
        )

    st.markdown(
        """
### 6 · Explainability methodology
Grad-CAM (Selvaraju et al., ICCV 2017) on the last convolutional block (`features[-1]`, 7×7 grid):
class-specific gradients are global-average-pooled into channel weights, the weighted sum is ReLU'd and
bilinearly upsampled to 224×224 (`src/explainability/gradcam.py`). The map is qualitative; correctness of the
reasoning is **not** implied.

### 7 · Calibration methodology
`src/evaluation/calibration.py` measures ECE (15 equal-width bins), NLL and the multiclass Brier score. A
single temperature is fitted by minimising **validation** NLL (deterministic grid search), then the calibrated
probabilities are evaluated on the untouched **test** split. Temperature scaling never changes predictions.

### 8 · Faithfulness experiments
`python -m src.experiments.gradcam_faithfulness` computes Grad-CAM **deletion** and **insertion** curves on the
validation split: pixels are perturbed in saliency order (replacement by per-image mean colour) and the
target-class probability is tracked; random pixel orders (multiple seeds) provide the baseline. Lower deletion
AUC / higher insertion AUC than random support - but do not prove - that the maps highlight evidence the model
uses."""
    )
    if faith:
        s = faith.get("summary", {})
        st.caption(
            f"Measured ({faith.get('n_images')} val images, {faith.get('random_baseline_seeds')} random seeds): "
            f"deletion AUC Grad-CAM {s.get('gradcam_deletion_auc')} vs random {s.get('random_deletion_auc')}; "
            f"insertion AUC {s.get('gradcam_insertion_auc')} vs random {s.get('random_insertion_auc')}."
        )

    st.markdown(
        """
### 9 · Reproducibility
Fixed seeds (split 123, app/experiments 42 unless stated), one config file (`configs/config.json`), strict
checkpoint validation with sha256, machine-readable reports under `outputs/reports/`, and a pytest suite with
synthetic fixtures (`tests/`) that verifies every code path without needing the real assets.

### 10 · Segmentation - and why there is none
The PBC dataset ships **no ground-truth segmentation masks**. Training a segmentation model on classical
pseudo-masks (Otsu/watershed) would only imitate that classical method and prove nothing, so no segmentation
accuracy is claimed. `src/segmentation/watershed.py` exists only as a classical image-processing utility used
by `src/experiments/segmentation_study.py` to *measure* whether centre-cell cropping changes classification
accuracy - an experiment, not a validated segmentation result.

### 11 · Limitations (summary)
Single dataset/acquisition setup; closed-set prediction (always one of 8 classes); unvalidated heuristic
image-quality warnings; coarse qualitative explanations; historical 97.07 % accuracy not reproduced;
research/educational use only - **not a clinical device**.
"""
    )

# Research notes: reference paper and feasible improvements

## Reference paper: what was and was not read

"Efficient blood cell classification from microscopic smear images using U-Net segmentation and a lightweight CNN"
(Scientific Reports, 2025; arXiv preprint 2405.14875 "BloodCell-Net"). During preparation only the abstract-level
description was accessible (the full-text page was blocked by a bot check). From that: U-Net segmentation, watershed
separation of overlapping cells, a custom lightweight CNN (LWCNN), nine cell types, and a dataset that the authors say is
available on request. **The full text was not read.** The headline 97.10 % figure and the presence of Grad-CAM in the
paper were *not* verified here and should be checked by you against the PDF before being quoted.

Comparability: the paper uses nine classes on its own dataset; this project uses the eight-class public PBC set, a
different split, and a pretrained MobileNetV2. Accuracy values must not be presented as a head-to-head comparison.

## Improvements: rationale, feasibility, measurement

| # | Improvement | Status | How it is evaluated | Claim allowed |
|---|---|---|---|---|
| 1 | Reproducible baseline (fixed seeds, recorded split seed 123, config file, one eval command) | Implemented | `run_evaluation` reproduces metrics from checkpoint + `splits.csv` | Only after you run it; 97.07 % is historical |
| 2 | Grad-CAM in the website | Implemented | Unit-tested mechanics; `gradcam_faithfulness` deletion test vs random deletion | "Qualitative explanation"; faithfulness only if the deletion drop exceeds random |
| 3 | Per-class P/R/F1, confusion matrix, top confusions, error CSV | Implemented | `run_evaluation` | Report values it prints |
| 4 | Size, params, CPU latency, throughput | Implemented | `benchmark` | Hardware-specific; state the CPU |
| 5 | Calibration (ECE/NLL, temperature scaling on val) | Implemented | `run_evaluation --calibrate` reports before/after on test | Probabilities are better calibrated only if ECE/NLL drop; accuracy is unchanged by construction |
| 6 | Robustness (blur, noise, JPEG, brightness, contrast, downscale) and OOD probe | Implemented | `robustness`; OOD AUROC of max-softmax | Only dataset-specific degradation numbers; AUROC near 0.5 means confidence is not an OOD detector |
| 7 | Watershed cell separation | Implemented as classical module + study | `segmentation_study`: baseline vs centre-crop accuracy, objects per image, multi-object subset | **No accuracy claim** until the study shows one on test with parameters fixed on val |
| 8 | U-Net segmentation | **Not implemented** | - | The PBC set has no ground-truth masks. Training on Otsu/watershed pseudo-masks would only imitate the classical method and prove nothing; it needs hand-annotated masks (a few hundred) to be meaningful |

## Protocol discipline

Tune any threshold/parameter on `--split val` only; run the final number once on `--split test`. Do not change the test set.
Note PBC images are one-cell-centred crops, so watershed has little to separate in most images; the study quantifies this
instead of assuming it.

## Heuristic quality warnings

`src/inference/quality.py` thresholds (brightness, contrast, blur, saturation, aspect) are unvalidated heuristics. They can be
calibrated with `compute_quality_stats` on the real dataset; until then they are warnings, not detectors.

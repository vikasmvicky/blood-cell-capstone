# YOLO11 Data Decision — Where Genuine Bounding-Box Annotations Come From

Date: 2026-10-22 · Branch: `feat/yolo11-detection` · Status: **decision recorded; external-dataset path rejected, CVAT pilot confirmed as primary path**

## 1. Question

Object detection needs genuine bounding-box annotations. The PBC dataset in
`data/raw/` has image-level class labels only (verified: zero YOLO label files
anywhere in `data/annotations/`; the existing classifier checkpoints are
whole-image models). This document records the investigation of legitimate
external detection datasets before committing to human annotation.

## 2. Candidates investigated

| Candidate | What it actually is | Verdict |
|---|---|---|
| "BBBC073" (Broad Institute) | **Does not exist.** No such accession appears in BBBC search results or Google results for the exact string. An earlier assumption that a Broad-hosted 8-class re-annotation of the PBC dataset existed was **wrong** and is retracted. | Rejected — nonexistent |
| TXL-PBC (arXiv 2407.13214) | Curated re-annotation of four public resources; 1,260 images, 18,143 boxes — but only **3 classes** (WBC / RBC / platelet) | Rejected — class taxonomy incompatible; mapping 3 super-classes onto our 8 PBC classes is impossible (WBC would need splitting into 5 subtypes) |
| BCCD (Kaggle) | 364 images, 3 classes (WBC / RBC / platelet), different acquisition source | Rejected — same taxonomy mismatch; also tiny and from different imaging equipment |
| Roboflow Universe "Blood_Cell_Classification" etc. | Community datasets, mostly 2-3 classes (WBC/RBC/platelet), mixed provenance, licenses vary per project | Rejected — no 8-class PBC-compatible option found; provenance/licensing not verifiable to publication standard |

No legitimate public dataset provides genuine bounding boxes for the full
8-class PBC taxonomy (basophil, eosinophil, erythroblast, ig, lymphocyte,
monocyte, neutrophil, platelet).

## 3. Decision

**Primary path: annotate the prepared 160-image pilot in CVAT** (user-approved
2026-10-22, "Hybrid both" option), per:

- `data/annotations/yolo11_pilot/` — 160 hash-verified image copies, fixed
  train/val/test splits (112/24/24) inherited from `outputs/reports/splits.csv`
- `docs/YOLO11_ANNOTATION_POLICY.md` — box rules, RBC-as-background policy
- `docs/YOLO11_CVAT_WORKFLOW.md` — CVAT project setup and export procedure
- `src/annotation/validate_labels.py` — acceptance validator (verified working
  2026-10-22: exits 2 with a clear message while labels are not yet exported)

The feasibility audit (`outputs/reports/yolo11_annotation_feasibility/feasibility_report.md`)
verified 30/30 sampled images are boxable.

## 4. Consequences and honest limitations

- **No training results can exist until the human annotation step is done.**
  Any YOLO11 metric claimed before validated labels exist would be fabricated.
  The pipeline (data validation, training config, inference, web app) will be
  implemented and tested with clearly-labelled synthetic fixtures where needed,
  and real training starts only after `validate_labels.py` passes.
- 160 images with ~1 box/image is a *small* detection dataset. This limits
  achievable mAP and must be stated as a limitation in the final report; the
  pilot's role is to establish a **measured, honest baseline** (Objective 1),
  with scale-up (more annotated images) as an explicit improvement lever
  (Objectives 3/5).
- The image-level class folder of each pilot image is a strong prior for the
  dominant cell's box label but is never treated as box ground truth
  (policy §1), so no boxes are ever invented from classification labels.

## 5. What would reopen this decision

- A classmate completing CVAT annotations for a larger set (then validate +
  scale, same policy).
- A future legitimate 8-class PBC bounding-box release (none found as of
  2026-10-22).

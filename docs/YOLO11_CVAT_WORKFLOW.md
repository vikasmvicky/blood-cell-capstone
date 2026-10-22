# YOLO11 CVAT Workflow — Annotation Pilot

Status: draft for pilot · Date: 2026-10-21 · Scope: get the 160 pilot images annotated in CVAT and validated locally · Companions: `docs/YOLO11_ANNOTATION_POLICY.md` (how to draw each box), `data/annotations/yolo11_pilot/README.md` (workspace + review log), `src/annotation/validate_labels.py` (acceptance check)

This is the operational workflow: CVAT project setup, the label/attribute schema CVAT must encode, export, and validation. It creates no labels by itself — every box is drawn by a human. No training happens at any point in this workflow.

## 1. One-time CVAT project setup

1. Create a CVAT project named `blood-cell-yolo11-pilot` (local CVAT instance or app.cvat.ai — either is fine; the export step is identical).
2. Add **exactly these 8 labels**, in this order, with these exact names (this order becomes the YOLO class-ID order in the export; `outputs/reports/class_to_idx.json` is the source of truth and must match):

   | Order | Label name | ID |
   |---|---|---|
   | 1 | `basophil` | 0 |
   | 2 | `eosinophil` | 1 |
   | 3 | `erythroblast` | 2 |
   | 4 | `ig` | 3 |
   | 5 | `lymphocyte` | 4 |
   | 6 | `monocyte` | 5 |
   | 7 | `neutrophil` | 6 |
   | 8 | `platelet` | 7 |

   Do not add any other label. In particular there is **no `rbc`, `unknown`, or `artifact` label** — RBCs and debris are unannotated background by policy (§3).
3. Add two **shape attributes** available to every label:
   - `confidence`: type radio, values `high` | `medium` | `low`, default `high` (policy §5). `medium` = class or extent uncertain.
   - `truncated`: type checkbox, default false (policy §7). Tick it for cells cut by an image edge.
4. Optionally create a CVAT user for the annotator and a separate reviewer account for the review pass (§5).

## 2. Create the task

1. New task named `blood-cell-yolo11-pilot-160`, project from step 1.
2. Upload **all 160 files** from `data/annotations/yolo11_pilot/images/` (that folder contains only the pilot copies — never upload the full dataset).
3. Set task layout to "Default", segmentation method "Annotation" (pre-annotation is intentionally not used — no model output may seed boxes).
4. After creation, verify the task shows 160 images, then **spot-check 3 random frames** against `pilot_manifest.csv` (`class_name` column) so the upload did not reorder or drop files.

## 3. Annotating

1. Read `docs/YOLO11_ANNOTATION_POLICY.md` fully before drawing anything. It defines the label set (§1), universal box rules (§2), the RBC background policy (§3), per-class drawing guidance with real feasibility examples (§4), ambiguity handling (§5), overlaps (§6), truncation (§7), minimum platelet size (§8), empty images (§9), and split preservation (§10).
2. Draw one box per visible instance of the 8 classes (policy §2.2) — most images have one dominant cell, but 8/30 feasibility-sampled images also contained extra platelets; annotate **all** of them.
3. Set `confidence` and `truncated` on every shape as you go. Never guess a class for an uncertain object; use a frame comment plus a `review_log.csv` row instead (policy §5).
4. Maintain the review log **while annotating**: append a row to `data/annotations/yolo11_pilot/review_log.csv` for every `truncated` shape, every `medium`/`low`-confidence shape, every deliberately unlabeled ambiguous object, and every empty image (schema in workspace README §3).

## 4. Work order and splits

- Annotate in manifest order (`pilot_index` 1…160) or by class — order does not matter, because each image's `train`/`val`/`test` assignment lives in `pilot_manifest.csv` and never changes (policy §10). CVAT itself is not the split authority.
- If a second annotator pass is agreed after the ambiguity readout (policy §5), re-annotate a stratified 10–15% subset — same images, same policy, no peeking at the first pass.

## 5. Review pass

1. A second person (or the same annotator after a cool-down, if no reviewer is available — record which) reviews every frame in CVAT.
2. Reviewer checks against the policy: tightness (§2.1), no RBC boxes (§3), per-class plausibility (§4), all instances present (§2.2), attributes correct (§5/§7), minimum platelet size respected (§8).
3. Disagreements are resolved by editing the shape in CVAT, with the decision noted in `review_log.csv`.

## 6. Export

1. In CVAT: project → **Export job data / Export dataset → YOLO 1.1** format (bounding boxes only; include "images" unchecked — the workspace already has byte-identical image copies, so export **annotations only**).
2. The export archive contains one `.txt` per image plus a `obj.names`/schema file. The label-file class order must match §1's table; **do not trust the export blindly** — the validator checks every class ID against `outputs/reports/class_to_idx.json`, not against export order (see module docstring of `src/annotation/validate_labels.py`).
3. Unzip the label files into `data/annotations/yolo11_pilot/labels/` so each `images/<stem>.jpg` has a sibling `labels/<stem>.txt`. Do not edit the `.txt` files by hand to "fix" validation errors — fix in CVAT and re-export.

## 7. Validate locally

From the project root (Windows):

```powershell
.venv\Scripts\python.exe -m src.annotation.validate_labels
```

Expected statuses per image: `ok` (≥1 well-formed line), `unverified_empty` (empty file — log it, policy §9), `pending_missing` (not yet annotated/exported), `invalid` (malformed line — must be zero before acceptance). Exit code 0 = no invalid lines; 1 = invalid lines present; 2 = setup error (missing manifest/class map/labels dir).

Optional per-image CSV report: `--report outputs/reports/yolo11_annotation_feasibility/validator_report.csv` (or any path you choose).

## 8. Acceptance criteria for the pilot

- `labels/` contains 160 files (or the shortfall is explicitly listed as `pending_missing`).
- Validator exit code 0; `invalid = 0`.
- Every `unverified_empty` image has a matching `review_log.csv` row.
- Ambiguity readout (counts by `confidence` and log issues) written up so the scale-up decision (double-annotation needed or not) can be made with evidence.

## 9. Out of scope (explicitly)

- No YOLO11 training, no fine-tuning, no mAP runs in this workflow — training is a later phase gated on this pilot's agreement statistics.
- No changes to `app_ui/`, `app.py`, or any application component.
- No modification of `outputs/reports/splits.csv`, `class_to_idx.json`, or the source dataset.

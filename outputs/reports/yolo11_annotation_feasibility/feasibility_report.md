# YOLO11 Annotation Feasibility Report

Date: 2026-10-20 · Project: `D:\blood-cell-capstone` · Status: read-only feasibility check (no labels created, no training, no dataset or application changes)

## 1. Inputs read (unchanged)

| File | Content |
|---|---|
| `outputs/reports/splits.csv` | 17,090 usable images; columns include `rel_path`, `class_name`, `width`, `height`, `sha256`, `split` (seed 123, 70/15/15) |
| `outputs/reports/image_manifest.csv` | 17,092 rows including 2 excluded images |
| `outputs/reports/class_to_idx.json` | 8 classes: basophil=0, eosinophil=1, erythroblast=2, ig=3, lymphocyte=4, monocyte=5, neutrophil=6, platelet=7 |
| `outputs/reports/split_integrity.json` | PASS; no duplicate leakage; counts match |

## 2. Reproducible sample selection

30 images selected deterministically from `splits.csv` (no RNG): classes in alphabetical order; per-class counts `[4,4,4,4,4,4,3,3]`; within each class, images sorted by `rel_path` and evenly spaced indices `round(i*(n-1)/(k-1))` taken (includes first and last). Full selection with SHA-256 hashes: `sample_manifest.csv`. Selection script: `select_and_sheet.py` (rerunnable; refuses to overwrite existing outputs).

Class coverage: basophil 4, eosinophil 4, erythroblast 4, ig 4, lymphocyte 4, monocyte 4, neutrophil 3, platelet 3. Splits represented: train 21, val 3, test 6. All images ~360x363 px JPEG.

## 3. Artifacts produced (new directory only)

- `outputs/reports/yolo11_annotation_feasibility/contact_sheet.png` — 6x5 grid, 2160x1955 px, each tile labelled with index, class, split and filename.
- `outputs/reports/yolo11_annotation_feasibility/contact_sheet_preview.jpg` — downscaled preview (1400 px).
- `outputs/reports/yolo11_annotation_feasibility/sample_manifest.csv` — the 30 selected images.
- `outputs/reports/yolo11_annotation_feasibility/per_image_assessment.csv` — per-image four-criterion assessment.
- `outputs/reports/yolo11_annotation_feasibility/select_and_sheet.py` — the selection/sheet script.

## 4. What the images show

Every sampled image is a cropped microscope field containing:

1. **One dominant nucleated cell** of the labelled class, usually centered, typically 90-250 px across (platelet-class images contain a single small purple platelet ~40 px instead).
2. **Many red blood cells** (typically 10-30 per field) — a ninth cell type that is *not* part of the 8-class label set.
3. Occasionally **extra small purple objects** (platelets or cell fragments) beyond the dominant cell — seen in at least 8 of 30 images (e.g. tiles 1, 17, 18, 19, 25, 27, 29, 30).

## 5. Assessment results (30/30 inspected at full resolution)

| Criterion | Result |
|---|---|
| Complete cell with visible background around boundary | 28/30 clearly complete; 1 ambiguous (tile 21, cytoplasm approaches bottom edge); 1 truncated (tile 6, EO_405046 — nucleus/cytoplasm cut at right edge) |
| Cell cropped at/touching image edges | 1 definite (tile 6), 1 ambiguous (tile 21) |
| Multiple cells or other objects | 30/30 contain many RBCs; 8/30 contain additional platelet-class objects; 1 image (tile 29) plausibly contains two platelet instances |
| Image artifacts or ambiguous boundaries | 6 images with notable ambiguity: tile 6 (low pink-on-pink contrast), 13 (background stain gradient), 17 (platelet attached to cell border), 18 and 22 (ragged/smudge-like borders), 21 and 23 (irregular, faint monocyte cytoplasm with vacuoles and RBC overlap) |

**Boxability verdict: 30/30 images can receive a meaningful bounding box around the dominant labelled cell**, with an explicit policy for truncated boxes (tile 6) and fuzzy cytoplasmic boundaries (mainly monocytes, tiles 21-23).

## 6. Can YOLO11 be trained on this dataset after annotation?

Yes, with caveats. Each image supports exactly one easy box for the dominant cell, so a single-instance-per-image detection dataset is feasible. Two structural caveats:

1. **Unlabelled RBCs.** YOLO treats unannotated objects as background. If RBCs are left unannotated, the model is penalised for detecting them and learns to suppress a very salient cell type. Annotators must either (a) add an `rbc` class and annotate all RBCs, or (b) document an ignore-region policy. This is the single most important annotation decision.
2. **Small objects.** Platelet boxes are ~40 px at native 360x363 resolution; training at `imgsz=640` upscales, which is acceptable, but very small or faint platelets need a consistent minimum-size rule and annotator examples.

Other requirements: annotate *all* instances of the 8 classes (multi-platelet images occur); use a truncated-box convention for edge-cut cells; double-annotate a 10-15% subset (oversampling monocytes and IG) to measure inter-annotator agreement before scaling.

## 7. Recommendation: **Path A — annotate a pilot subset of the existing dataset**

The images are suitable for meaningful bounding-box annotation: consistent single-cell crops, centred subjects, strong stain contrast for nuclei, no blocking artifacts. A pilot of ~500-1,000 images (e.g. 300 train / 100 val / 100 test, drawn from the existing leakage-safe `splits.csv` boundaries) is the next step. Path B (an external multi-cell detection dataset) is not required for feasibility, but could later be added as an accelerator or for pretraining if pilot annotation agreement proves poor.

### Suggested pilot annotation policy (for the annotators)

- Label all instances of the 8 classes per image; decide the RBC policy first (recommend `rbc` class or explicit ignore regions).
- Clip boxes at image edges for truncated cells (record `truncated=true` in the tool).
- Platelet: annotate every visible platelet above the agreed minimum size (~25 px).
- Boundary convention for fuzzy cytoplasm (monocyte/IG): box the visible cytoplasm extent, not the halo; provide the ambiguity examples from this study (tiles 21, 22, 23, 18) in the guideline.
- Keep class mapping identical to `class_to_idx.json`; the `ig` class merges MY/MMY/PMY prefixes and neutrophil merges BNE/SNE/NEUTROPHIL prefixes — do not split subtypes unless the label set is revised.

## 8. Limitations

- 30 images out of 17,090 were inspected; the selection is deterministic and covers all classes and all splits, but rare failure modes (heavy artifact slides, badly out-of-focus crops) may be under-sampled.
- Assessments are a single human-equivalent visual review; no second annotator and no quantitative boundary metric was used.
- Feasibility was judged on the *existing* 360x363 crops; whole-slide or larger-field images were not evaluated.
- Environment note for later training: Python 3.11.5, torch 2.14.1+cpu, CUDA unavailable, `ultralytics` not installed — a full YOLO11 training run on this machine would be CPU-only and slow.
- No labels, models or predictions were created; no files outside this new directory were modified.

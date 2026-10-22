# YOLO11 Annotation Policy — Blood-Cell Detection Pilot

Status: draft for pilot · Scope: `data/annotations/yolo11_pilot/` (160 images) · Companions: `data/annotations/yolo11_pilot/README.md`, `docs/YOLO11_CVAT_WORKFLOW.md`

This policy defines how to draw bounding boxes for the pilot so that annotators produce boxes a model can learn from consistently. Nothing here creates labels; all boxes are drawn by a human in CVAT.

## 1. Label set and class IDs

Class IDs are fixed by `outputs/reports/class_to_idx.json` and must not be renumbered or extended:

| ID | Class name (folder) | Display name | In this pilot |
|---|---|---|---|
| 0 | `basophil` | Basophil | nucleated granulocyte |
| 1 | `eosinophil` | Eosinophil | nucleated granulocyte |
| 2 | `erythroblast` | Erythroblast | nucleated red-cell precursor |
| 3 | `ig` | Immature granulocyte (IG) | myelocyte / metamyelocyte / promyelocyte |
| 4 | `lymphocyte` | Lymphocyte | mononuclear |
| 5 | `monocyte` | Monocyte | mononuclear, large |
| 6 | `neutrophil` | Neutrophil | segmented or band |
| 7 | `platelet` | Platelet | small anuclear fragment |

- The `ig` class **merges** source filename prefixes `MY_`, `MMY_`, `PMY_`; `neutrophil` merges `BNE_`, `SNE_`, `NEUTROPHIL_`. Do not split or rename subtypes.
- Each pilot image comes from one class folder; that image-level label is a strong prior but **not** ground truth for boxes. Draw what is visible.

## 2. Universal box rules

1. **Box the object, not the crop**: tight box around the visible extent of the cell (nucleus + cytoplasm for leukocytes; the fragment itself for platelets). Include the cytoplasm rim; exclude halo, stain halo, and empty background.
2. **One box per visible instance** of any of the 8 classes. If a second target-class instance is visible (most often extra platelets), annotate it too.
3. **Every drawn box must have `x_center, y_center, width, height` all strictly inside (0, 1)** after YOLO normalization; positive width and height. Boxes are clipped at image edges by the tool.
4. **Truncated cells** (touching/cut by an image edge): draw the box around the visible portion, clipped at the edge; in CVAT tick **occluded/outside** as applicable and record `truncated=true` in the review log (see workspace README for the log file).
5. **Confidence field**: if you are not sure an object belongs to a class, do not guess — see §5.
6. **No invented biology**: a box gets a class only if the object's appearance matches the class description in §4. Do not assign a label to an uncertain object to "fill" the image.

## 3. Red blood cells (RBCs) and other background objects

1. **Never label an RBC as any of the eight classes.** An erythrocyte (pink biconcave disc, no nucleus) is not an `erythroblast` (nucleated precursor) and not a `platelet`.
2. **RBCs are ignored background in this pilot.** Draw no box on them, and do not use CVAT's "ignore region" attribute — the exported YOLO format has no ignore construct, so an unannotated RBC is simply background. Annotators must still *look at* overlapping RBCs to decide boundaries for target cells (§6), but no RBC box is ever created.
3. **When an RBC is adjacent to a target cell**: box only the target cell up to its visible boundary; where the RBC occludes part of the cell, end the box at the occluder's edge only if the cell body is genuinely cut — otherwise include the full cell extent (the occluded part is still inside the box).
4. **Other background objects** (stain debris, dust, background streaks, smear-edge gradients): no labels, no "artifact" or "unknown" classes may be added.
5. **Unannotated is not "negative".** Because RBCs and debris are intentionally unlabeled, an image with zero label lines must not be treated as a verified negative example of all classes — see `src/annotation/validate_labels.py`, which reports empty label files as `unverified_empty`, never as verified negatives.

## 4. Per-class guidance

- **0 basophil** — coarse dark-purple cytoplasmic granules often partially obscuring the nucleus; box nucleus + granular cytoplasm. Loose edge granules count as cell; do not extend the box into nearby platelet dots.
- **1 eosinophil** — bilobed nucleus with pink-red granular cytoplasm. Pink granules next to pink RBCs can be low-contrast: trace the outer cytoplasm edge carefully; when cut by the frame, §2.4 applies (feasibility example: `eosinophil/EO_405046.jpg`).
- **2 erythroblast** — dense round dark nucleus with a thin blue cytoplasm rim (may be pale or partially shed). Box the full rim where visible; if the rim is only faintly visible, box to the nucleus-plus-halo boundary consistently and note it in the review log.
- **3 ig** — immature neutrophil-lineage cells (myelocyte/metamyelocyte/promyelocyte): rounded or indented nucleus, cytoplasm less pale than a segmented neutrophil. If a cell genuinely looks mature/segmented, prefer `neutrophil`; if genuinely immature, `ig`. Ambiguity between `ig` and `neutrophil` is expected — log it (§5).
- **4 lymphocyte** — small-to-medium round cell, dense dark nucleus, scant blue cytoplasm. Ragged/smudged borders: box the visible cell extent, not the frayed fringe tail beyond it.
- **5 monocyte** — largest cell; kidney/irregular nucleus, abundant pale gray-blue cytoplasm, vacuoles common; borders often fuzzy (feasibility examples: `monocyte/MO_389525.jpg`, `MO_100695.jpg`). Box the visible cytoplasm extent, **not** the smear halo; if truly unclear where the cell ends, log ambiguity rather than inflating the box.
- **6 neutrophil** — segmented (2-5 lobes) or band nucleus, pale granular cytoplasm. Pale staining is still a neutrophil; do not shrink the box to the nucleus only.
- **7 platelet** — small purple fragment, roughly 15-60 px at native 360x363. **Minimum box size: ~10 px (~3% of image width)**; smaller faint specks are left unlabeled (per §3.4 background). Annotate **all** platelets at or above the minimum, not just the darkest one — multiple instances per image are common (feasibility: `platelet/PLATELET_532560.jpg` shows two).

## 5. Ambiguous objects

- Use the confidence attribute in CVAT: every shape gets `confidence = high | medium | low` (`medium` = class or extent uncertain, e.g. `ig` vs `neutrophil`, faint erythroblast rim, monocyte halo).
- **Do not invent a class for an uncertain object.** If it cannot be matched to §4 with medium confidence, draw nothing and add a comment on the frame with your tentative observation (free text, not a class).
- Ambiguity stats from the pilot are read out of the exported labels + review log to decide whether a second annotator pass is needed before scaling.

## 6. Overlapping objects

- Nucleated cell overlapping an RBC: cell box follows the cell's visible boundary; RBC gets nothing (§3).
- Target cell touching/overlapping another target cell (e.g. platelet overlapping a leukocyte border, feasibility `lymphocyte/LY_100275.jpg`): two separate boxes; each box covers its own object's extent; overlap between boxes is allowed.
- Platelet directly adherent to a leukocyte: label the platelet `platelet` only if it is clearly a separate fragment with its own outline; otherwise leave unlabeled and comment.

## 7. Truncated cells at image edges

- Visible portion only, clipped by the frame (feasibility example: `eosinophil/EO_405046.jpg`, right edge).
- Mark the shape with CVAT attribute `truncated = true`.
- Never extend a box beyond the image and never exclude an edge-cut cell from labeling.

## 8. Very small platelets

- Below the ~10 px / 3%-of-width minimum: leave unlabeled (background per §3.4); do not "best-guess" a box.
- At or above the minimum but diffuse: box the visible fragment; set `confidence = low` if the extent is unclear.

## 9. Empty images

- If, after review, an image truly contains none of the 8 classes (unexpected for this pilot, since every image was selected from a class folder), leave the label file empty and record it in the review log as **unverified** — empty never counts as a verified negative (§3.5).

## 10. Split preservation

- Every pilot image keeps its original `train`/`val`/`test` assignment from `outputs/reports/splits.csv`, recorded in `pilot_manifest.csv`. Annotation order must not change splits, and an image must never be moved between splits after labeling.

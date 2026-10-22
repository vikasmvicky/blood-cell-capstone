"""YOLO11 annotation feasibility — reproducible sample selection + contact sheet.

Read-only with respect to the dataset and all pre-existing project files:
reads `outputs/reports/splits.csv`, reads the selected JPEGs, and writes
ONLY into this directory (`outputs/reports/yolo11_annotation_feasibility/`).
Refuses to overwrite any existing file.

Selection rule (deterministic, no RNG):
  1. Read splits.csv; keep rows with readable == "True".
  2. Group by class_name; sort each group by rel_path (ascending).
  3. Classes in alphabetical order; per-class sample counts =
     [4, 4, 4, 4, 4, 4, 3, 3] (total 30).
  4. Within a class of n images, the k sampled indices are
     idx_i = round(i * (n - 1) / (k - 1)) for i in 0..k-1 (evenly spaced,
     includes first and last).

Run from the project root:
  .venv/Scripts/python.exe outputs/reports/yolo11_annotation_feasibility/select_and_sheet.py
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS_CSV = PROJECT_ROOT / "outputs" / "reports" / "splits.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "yolo11_annotation_feasibility"
SHEET_PATH = OUT_DIR / "contact_sheet.png"
MANIFEST_PATH = OUT_DIR / "sample_manifest.csv"

PER_CLASS_COUNTS = {
    "basophil": 4,
    "eosinophil": 4,
    "erythroblast": 4,
    "ig": 4,
    "lymphocyte": 4,
    "monocyte": 4,
    "neutrophil": 3,
    "platelet": 3,
}
COLUMNS = 6
TILE_W, TILE_H = 360, 363
LABEL_H = 34


def evenly_spaced(n: int, k: int) -> list[int]:
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def main() -> int:
    for p in (SHEET_PATH, MANIFEST_PATH):
        if p.exists():
            print(f"REFUSING TO OVERWRITE existing file: {p}", file=sys.stderr)
            return 1

    with SPLITS_CSV.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["readable"] == "True"]

    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["class_name"], []).append(r)

    missing = set(PER_CLASS_COUNTS) - set(by_class)
    if missing:
        print(f"ERROR: classes absent from splits.csv: {sorted(missing)}", file=sys.stderr)
        return 1

    selected: list[dict] = []  # tile order = class alphabetical, then index
    for cls in sorted(PER_CLASS_COUNTS):
        group = sorted(by_class[cls], key=lambda r: r["rel_path"])
        k = PER_CLASS_COUNTS[cls]
        for idx in evenly_spaced(len(group), k):
            row = group[idx]
            selected.append(
                {
                    "tile": len(selected) + 1,
                    "class_name": cls,
                    "sample_index_in_class": idx,
                    "class_size": len(group),
                    "filename": row["filename"],
                    "rel_path": row["rel_path"],
                    "split": row["split"],
                    "width": row["width"],
                    "height": row["height"],
                    "sha256": row["sha256"],
                }
            )

    if len(selected) != 30:
        print(f"ERROR: expected 30 selected images, got {len(selected)}", file=sys.stderr)
        return 1

    # Verify all selected files exist before building the sheet.
    paths = []
    for s in selected:
        p = PROJECT_ROOT / "data" / "raw" / "PBC_dataset_normal_DIB" / Path(s["rel_path"])
        if not p.is_file():
            print(f"ERROR: missing image on disk: {p}", file=sys.stderr)
            return 1
        paths.append(p)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0]))
        w.writeheader()
        w.writerows(selected)

    rows_grid = (len(selected) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGB", (COLUMNS * TILE_W, rows_grid * (TILE_H + LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for s, p in zip(selected, paths):
        t = s["tile"] - 1
        cx = (t % COLUMNS) * TILE_W
        cy = (t // COLUMNS) * (TILE_H + LABEL_H)
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.size != (TILE_W, TILE_H):
                im = im.resize((TILE_W, TILE_H))
            sheet.paste(im, (cx, cy + LABEL_H))
        label = f"#{s['tile']} {s['class_name']} [{s['split']}]"
        draw.text((cx + 4, cy + 2), label, fill="black", font=font)
        draw.text((cx + 4, cy + 18), s["filename"], fill="black", font=font)
        draw.rectangle(
            [cx, cy + LABEL_H, cx + TILE_W - 1, cy + LABEL_H + TILE_H - 1],
            outline="gray",
        )

    sheet.save(SHEET_PATH)

    print(f"selected={len(selected)}")
    for cls in sorted(PER_CLASS_COUNTS):
        n = sum(1 for s in selected if s["class_name"] == cls)
        print(f"  {cls}: {n}")
    print(f"contact_sheet: {SHEET_PATH}")
    print(f"manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

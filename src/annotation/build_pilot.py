"""Build the YOLO11 annotation pilot manifest and image workspace.

Reads the existing leakage-safe split manifest (`outputs/reports/splits.csv`)
and class map (`outputs/reports/class_to_idx.json`) without modifying them,
verifies the recorded split assignments and image-level labels, selects a
reproducible stratified pilot of up to 160 images (20 per class), copies just
those images into `data/annotations/yolo11_pilot/images/` (CVAT needs file
uploads; the full dataset is NOT copied), and writes `pilot_manifest.csv` +
`pilot_selection_summary.json` into the workspace.

Selection rule (deterministic, no RNG):
  1. Verify splits.csv: readable == True, rel_path unique, folder in rel_path
     == class_name == class, class in class_to_idx.
  2. Deduplicate: drop every row whose sha256 also occurs on another row
     (duplicate images are excluded entirely, so no image can enter twice and
     no duplicate pair can leak across pilot splits).
  3. Stratify by (class_name, split); per class take
     train=14, val=3, test=3 (= 20/class, 160 total), preserving each image's
     original split assignment.
  4. Within a stratum sorted by rel_path, take k evenly spaced indices
     idx_i = round(i * (n - 1) / (k - 1)) (includes first and last).
  5. Verify each selected file on disk: exists, sha256 matches the manifest,
     PIL can open it, and its size matches the recorded width/height.
  6. Copy to the workspace with an overwrite-refusal check, then re-hash the
     copy to confirm it matches the source.

Run from the project root:
  .venv/Scripts/python.exe -m src.annotation.build_pilot
"""

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_CSV = PROJECT_ROOT / "outputs" / "reports" / "splits.csv"
CLASS_MAP = PROJECT_ROOT / "outputs" / "reports" / "class_to_idx.json"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "PBC_dataset_normal_DIB"
WORKSPACE = PROJECT_ROOT / "data" / "annotations" / "yolo11_pilot"
IMAGES_DIR = WORKSPACE / "images"
MANIFEST_OUT = WORKSPACE / "pilot_manifest.csv"
SUMMARY_OUT = WORKSPACE / "pilot_selection_summary.json"

PER_CLASS_SPLIT_COUNTS = {"train": 14, "val": 3, "test": 3}
EXPECTED_PER_CLASS = sum(PER_CLASS_SPLIT_COUNTS.values())  # 20
EXPECTED_TOTAL = EXPECTED_PER_CLASS * 8  # 160


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evenly_spaced(n: int, k: int) -> list[int]:
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    for p in (MANIFEST_OUT, SUMMARY_OUT):
        if p.exists():
            return fail(f"REFUSING TO OVERWRITE existing file: {p}")
    if IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()):
        return fail(f"REFUSING TO OVERWRITE: {IMAGES_DIR} is not empty")

    with CLASS_MAP.open(encoding="utf-8") as f:
        class_to_idx = json.load(f)

    with SPLITS_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return fail("splits.csv is empty")

    # --- Verification: labels and split assignments must be verifiable. ---
    problems = []
    seen_paths = set()
    for i, r in enumerate(rows, start=2):  # header is line 1
        if r["readable"] != "True":
            problems.append(f"line {i}: not readable")
        if r["rel_path"] in seen_paths:
            problems.append(f"line {i}: duplicate rel_path {r['rel_path']}")
        seen_paths.add(r["rel_path"])
        if r["class_name"] != r["class"] or r["rel_path"].split("/")[0] != r["class_name"]:
            problems.append(f"line {i}: label/path mismatch {r['rel_path']}")
        if r["class_name"] not in class_to_idx:
            problems.append(f"line {i}: unknown class {r['class_name']}")
        if r["split"] not in ("train", "val", "test"):
            problems.append(f"line {i}: bad split {r['split']}")
    if problems:
        for p in problems[:20]:
            print(p, file=sys.stderr)
        return fail(f"{len(problems)} verification problem(s) in splits.csv - stopping")

    # --- Deduplicate: exclude any image whose hash appears more than once. ---
    hash_counts = Counter(r["sha256"] for r in rows)
    dup_excluded = [r for r in rows if hash_counts[r["sha256"]] > 1]
    uniq = [r for r in rows if hash_counts[r["sha256"]] == 1]

    # --- Stratified deterministic selection. ---
    strata: dict[tuple[str, str], list[dict]] = {}
    for r in uniq:
        strata.setdefault((r["class_name"], r["split"]), []).append(r)

    selected: list[dict] = []
    for cls in sorted(class_to_idx, key=class_to_idx.get):  # class-id order
        for split in ("train", "val", "test"):
            need = PER_CLASS_SPLIT_COUNTS[split]
            group = sorted(strata.get((cls, split), []), key=lambda r: r["rel_path"])
            if len(group) < need:
                return fail(
                    f"stratum {cls}/{split} has {len(group)} unique images, need {need}"
                )
            for idx in evenly_spaced(len(group), need):
                selected.append(group[idx])

    if len(selected) != EXPECTED_TOTAL:
        return fail(f"expected {EXPECTED_TOTAL} selected, got {len(selected)}")
    if len({r["sha256"] for r in selected}) != len(selected):
        return fail("duplicate hashes within the selection")

    # --- Verify every selected file against the recorded manifest. ---
    records = []
    for r in selected:
        src = DATASET_ROOT / Path(r["rel_path"])
        if not src.is_file():
            return fail(f"missing on disk: {src}")
        actual = sha256_file(src)
        if actual != r["sha256"]:
            return fail(f"sha256 mismatch (label/path integrity): {r['rel_path']}")
        with Image.open(src) as im:
            im.load()
            if im.size != (int(r["width"]), int(r["height"])):
                return fail(
                    f"size mismatch for {r['rel_path']}: "
                    f"{im.size} != ({r['width']}, {r['height']})"
                )
        records.append(
            {
                "pilot_index": 0,  # filled below
                "class_name": r["class_name"],
                "class_id": class_to_idx[r["class_name"]],
                "split": r["split"],
                "filename": r["filename"],
                "source_rel_path": r["rel_path"],
                "workspace_image_path": f"data/annotations/yolo11_pilot/images/{r['filename']}",
                "width": r["width"],
                "height": r["height"],
                "sha256": r["sha256"],
            }
        )

    # Stable final ordering: class-id, then split order, then rel_path.
    split_order = {"train": 0, "val": 1, "test": 2}
    records.sort(key=lambda x: (x["class_id"], split_order[x["split"]], x["source_rel_path"]))
    for i, rec in enumerate(records, start=1):
        rec["pilot_index"] = i

    # --- Copy images (no moves; source dataset untouched), refuse overwrite. ---
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    names = [r["filename"] for r in records]
    if len(set(names)) != len(names):
        return fail("filename collision when flattening into workspace images/")

    copied = 0
    for rec in records:
        dst = IMAGES_DIR / rec["filename"]
        if dst.exists():
            return fail(f"REFUSING TO OVERWRITE existing file: {dst}")
        shutil.copy2(DATASET_ROOT / Path(rec["source_rel_path"]), dst)
        if sha256_file(dst) != rec["sha256"]:
            return fail(f"copy hash mismatch: {dst}")
        copied += 1

    # --- Write manifest + summary (refusal already checked above). ---
    with MANIFEST_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    class_counts = Counter(r["class_name"] for r in records)
    split_counts = Counter(r["split"] for r in records)
    summary = {
        "created_by": "src/annotation/build_pilot.py",
        "source_manifest": "outputs/reports/splits.csv",
        "source_manifest_modified": False,
        "selection_rule": (
            "deterministic; dedup by sha256 (all duplicate-hash rows excluded); "
            "stratified (class x original split) with train=14,val=3,test=3 per class; "
            "within stratum: sort by rel_path, evenly spaced indices incl. first/last"
        ),
        "per_class_split_counts": PER_CLASS_SPLIT_COUNTS,
        "total_images": len(records),
        "class_counts": dict(sorted(class_counts.items(), key=lambda kv: class_to_idx[kv[0]])),
        "split_counts": dict(split_counts),
        "duplicate_hash_rows_excluded_from_pool": len(dup_excluded),
        "verifications": {
            "labels_and_splits_verified": True,
            "selected_sha256_matches_source": True,
            "selected_dimensions_match_manifest": True,
            "copies_hash_verified": True,
        },
    }
    with SUMMARY_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"selected={len(records)} copied={copied}")
    print("class_counts:", json.dumps(summary["class_counts"]))
    print("split_counts:", json.dumps(split_counts))
    print(f"duplicate_hash_rows_excluded={len(dup_excluded)}")
    print(f"manifest: {MANIFEST_OUT}")
    print(f"summary: {SUMMARY_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

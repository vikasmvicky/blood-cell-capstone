"""Validate YOLO label files for the YOLO11 annotation pilot.

Reads `pilot_manifest.csv` and checks the exported label file of every pilot
image against the class map in `outputs/reports/class_to_idx.json`. The class
map is the single source of truth for class IDs: the validator never assumes
that the class order in a CVAT export (its ``obj.names`` / task label order)
matches the model's class IDs, so every label line is checked against the IDs
recorded in the manifest/class map, not against export order.

Per-image status (one of):
  ok               >= 1 well-formed label line.
  unverified_empty label file exists but has 0 lines. Because RBCs and debris
                   are intentionally unlabeled (policy section 3.5 / 9), an
                   empty label file is NOT a verified negative example of all
                   classes; it is reported as unverified and must be recorded
                   in the review log.
  pending_missing  no label file for a manifest image: annotation not done
                   yet (or the export is incomplete). Pending/unverified --
                   again, never treated as a verified negative.
  invalid          the label file contains at least one malformed line; every
                   offending line is reported with its reason.

Per-line checks (YOLO detection format: ``class_id x_center y_center width
height``, whitespace-separated, all coordinates normalized to [0, 1]):
  - exactly 5 fields
  - class_id is an integer present in class_to_idx.json
  - x_center, y_center, width, height parse as floats
  - 0 < x_center < 1, 0 < y_center < 1 (strictly inside, policy section 2.3)
  - 0 < width < 1, 0 < height < 1 (strictly positive -- a zero-width or
    zero-height box is invalid)

Exit codes: 0 = no invalid label lines (pending/unverified images are
reported but do not fail the run); 1 = at least one invalid line; 2 = setup
error (missing manifest / class map / labels dir).

Run from the project root, after dumping labels from CVAT into
`data/annotations/yolo11_pilot/labels/`:

  .venv/Scripts/python.exe -m src.annotation.validate_labels
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "annotations" / "yolo11_pilot"
DEFAULT_MANIFEST = DEFAULT_WORKSPACE / "pilot_manifest.csv"
DEFAULT_LABELS_DIR = DEFAULT_WORKSPACE / "labels"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "outputs" / "reports" / "class_to_idx.json"

STATUSES = ("ok", "unverified_empty", "pending_missing", "invalid")


def validate_line(
    line: str, valid_class_ids: set[int]
) -> list[str]:
    """Return a list of problems for one YOLO label line (empty == valid)."""
    fields = line.split()
    if len(fields) != 5:
        return [f"expected 5 fields, got {len(fields)}"]
    problems: list[str] = []
    class_str, *coord_strs = fields
    try:
        class_id = int(class_str)
    except ValueError:
        problems.append(f"class_id {class_str!r} is not an integer")
        class_id = None
    if class_id is not None and class_id not in valid_class_ids:
        problems.append(
            f"class_id {class_id} not in class map "
            f"(valid ids: {sorted(valid_class_ids)})"
        )
    coords: dict[str, float] = {}
    for name, s in zip(("x_center", "y_center", "width", "height"), coord_strs):
        try:
            coords[name] = float(s)
        except ValueError:
            problems.append(f"{name} {s!r} is not a float")
    for name in ("x_center", "y_center", "width", "height"):
        v = coords.get(name)
        if v is None:
            continue
        if not (0.0 < v < 1.0):
            problems.append(f"{name}={v} not strictly inside (0, 1)")
    return problems


def validate_image_label(
    label_path: Path, valid_class_ids: set[int]
) -> tuple[str, list[str]]:
    """Classify one label file; returns (status, line_problem_descriptions)."""
    if not label_path.is_file():
        return "pending_missing", []
    problems: list[str] = []
    text = label_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for lineno, line in enumerate(lines, start=1):
        for p in validate_line(line, valid_class_ids):
            problems.append(f"line {lineno}: {p} -- {line!r}")
    if not lines:
        return "unverified_empty", []
    if problems:
        return "invalid", problems
    return "ok", []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument(
        "--report", type=Path, default=None,
        help="optional path for a per-image CSV report",
    )
    args = parser.parse_args(argv)

    for p in (args.manifest, args.class_map):
        if not p.is_file():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 2
    if not args.labels_dir.is_dir():
        print(
            f"ERROR: labels dir not found: {args.labels_dir}\n"
            "Dump the CVAT annotations first (see docs/YOLO11_CVAT_WORKFLOW.md).",
            file=sys.stderr,
        )
        return 2

    with args.class_map.open(encoding="utf-8") as f:
        class_to_idx = json.load(f)
    valid_class_ids = set(class_to_idx.values())

    with args.manifest.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    if not manifest:
        print("ERROR: manifest is empty", file=sys.stderr)
        return 2

    rows: list[dict[str, str]] = []
    counts = dict.fromkeys(STATUSES, 0)
    for rec in manifest:
        stem = Path(rec["filename"]).stem
        status, problems = validate_image_label(
            args.labels_dir / f"{stem}.txt", valid_class_ids
        )
        counts[status] += 1
        rows.append(
            {
                "pilot_index": rec["pilot_index"],
                "filename": rec["filename"],
                "split": rec["split"],
                "class_name": rec["class_name"],
                "status": status,
                "n_problems": str(len(problems)),
                "problems": " | ".join(problems),
            }
        )
        marker = {"ok": "  ", "unverified_empty": "!!", "pending_missing": "..",
                  "invalid": "XX"}[status]
        print(f"[{marker}] {rec['filename']}: {status}")
        for p in problems:
            print(f"      {p}")

    manifest_stems = {Path(r["filename"]).stem for r in manifest}
    orphans = sorted(
        p.name for p in args.labels_dir.glob("*.txt")
        if p.stem not in manifest_stems
    )
    for name in orphans:
        print(f"WARNING: label file has no manifest image: {name}")

    print()
    print(f"images checked : {len(manifest)}")
    for status in STATUSES:
        print(f"{status:>16}: {counts[status]}")
    if orphans:
        print(f"{'orphan_labels':>16}: {len(orphans)} (warning only)")
    print()
    print(
        "NOTE: pending_missing and unverified_empty are PENDING/UNVERIFIED "
        "images, never verified negative examples of all classes "
        "(policy sections 3.5 and 9). Record them in the review log."
    )

    if args.report is not None:
        with args.report.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"report written: {args.report}")

    return 1 if counts["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

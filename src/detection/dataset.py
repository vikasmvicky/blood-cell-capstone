"""Validate the YOLO11 detection dataset (Stage 1) using genuine CVAT labels.

Reads the pilot workspace (``data/annotations/yolo11_pilot``) and checks that
every image is paired with a genuine YOLO-format label file exported from CVAT.
This module NEVER creates, infers or repairs bounding boxes: the image-level
``class_name``/``class_id`` columns in ``pilot_manifest.csv`` are annotator
priors only (policy section 1) and are never treated as box ground truth.

Checks performed:
  - image files exist, are readable, and match the manifest dimensions/hash
  - label files exist per image (missing ones are reported, never invented)
  - per-line YOLO format: 5 fields, class id in ``class_to_idx.json``,
    normalized coordinates strictly inside (0, 1), box inside the image
  - empty label files -> warning (``unverified_empty``, never a verified
    negative: policy sections 3.5 and 9)
  - duplicate image sha256 within the dataset and train/val/test split
    leakage (same content in more than one split)
  - orphan label files with no matching manifest image
Report: per-split image/box counts, per-class box counts, issues, warnings.

Exit codes (CLI): 0 = valid, 1 = issues found, 2 = setup error (annotations
not exported yet -- the message lists exactly what is missing and how to
export YOLO labels from CVAT, see docs/YOLO11_CVAT_WORKFLOW.md section 6).

Run from the project root:

  .venv/Scripts/python.exe -m src.detection.dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "annotations" / "yolo11_pilot"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "outputs" / "reports" / "class_to_idx.json"

VALID_SPLITS = ("train", "val", "test")
BOX_COORDS = ("x_center", "y_center", "width", "height")
# Tolerance for the "box fully inside the image" check (float round-off).
BOUNDS_EPS = 1e-6

CVAT_EXPORT_HINT = """\
No genuine YOLO bounding-box labels are available yet. Export them from CVAT
(never hand-create label files, never derive boxes from image-level labels):
  1. In CVAT, open project 'blood-cell-yolo11-pilot' and use
     Export job data / Export dataset -> format "YOLO 1.1", bounding boxes
     only, "include images" unchecked (docs/YOLO11_CVAT_WORKFLOW.md section 6).
  2. Unzip the archive so every images/<stem>.jpg has a sibling
     labels/<stem>.txt (one line per box: "class_id x_center y_center width
     height", coordinates normalized to [0, 1]).
  3. Re-run: .venv/Scripts/python.exe -m src.detection.dataset
Class IDs must match outputs/reports/class_to_idx.json (0 basophil ...
7 platelet); do not trust the export order blindly."""


class SetupError(RuntimeError):
    """Raised when the dataset workspace is not ready (missing dirs/files)."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ParsedLabel:
    """Result of parsing one YOLO label file."""

    n_boxes: int = 0  # fully valid lines only
    class_ids: list[int] = field(default_factory=list)  # valid lines only
    problems: list[str] = field(default_factory=list)  # per-line, with line no.


def parse_label_file(path: Path, valid_class_ids: set[int]) -> ParsedLabel:
    """Parse one YOLO detection label file; count only fully valid lines."""
    out = ParsedLabel()
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            out.problems.append(
                f"line {lineno}: expected 5 fields, got {len(fields)} -- {line!r}"
            )
            continue
        cls_str, *coord_strs = fields
        try:
            cls_id = int(cls_str)
        except ValueError:
            out.problems.append(
                f"line {lineno}: class_id {cls_str!r} is not an integer"
            )
            continue
        if cls_id not in valid_class_ids:
            out.problems.append(
                f"line {lineno}: class_id {cls_id} not in class map "
                f"(valid ids: {sorted(valid_class_ids)})"
            )
            continue
        try:
            x, y, w, h = (float(s) for s in coord_strs)
        except ValueError:
            bad = [
                s for s, name in zip(coord_strs, BOX_COORDS)
                if not _is_float(s)
            ]
            out.problems.append(
                f"line {lineno}: not a float: {', '.join(repr(b) for b in bad)}"
            )
            continue
        problems: list[str] = []
        for name, v in zip(BOX_COORDS, (x, y, w, h)):
            if not (0.0 < v < 1.0):
                problems.append(f"{name}={v} not strictly inside (0, 1)")
        if x - w / 2.0 < -BOUNDS_EPS or x + w / 2.0 > 1.0 + BOUNDS_EPS:
            problems.append(f"box extends outside image horizontally (x={x}, w={w})")
        if y - h / 2.0 < -BOUNDS_EPS or y + h / 2.0 > 1.0 + BOUNDS_EPS:
            problems.append(f"box extends outside image vertically (y={y}, h={h})")
        if problems:
            out.problems.append(
                f"line {lineno}: {'; '.join(problems)} -- {line!r}"
            )
            continue
        out.n_boxes += 1
        out.class_ids.append(cls_id)
    return out


def _is_float(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


@dataclass
class DatasetReport:
    """Aggregated validation result; ``issues`` fail, ``warnings`` inform."""

    workspace: Path
    class_to_idx: dict[str, int]
    images_checked: int = 0
    per_split_images: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(VALID_SPLITS, 0)
    )
    per_split_boxes: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(VALID_SPLITS, 0)
    )
    per_class_boxes: dict[str, int] = field(default_factory=dict)
    missing_labels: int = 0
    empty_labels: int = 0
    invalid_labels: int = 0
    duplicate_hash_groups: int = 0
    leakage_groups: int = 0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def to_json_dict(self) -> dict:
        return {
            "workspace": str(self.workspace),
            "class_to_idx": self.class_to_idx,
            "valid": self.is_valid,
            "images_checked": self.images_checked,
            "per_split_images": self.per_split_images,
            "per_split_boxes": self.per_split_boxes,
            "per_class_boxes": self.per_class_boxes,
            "missing_labels": self.missing_labels,
            "empty_labels": self.empty_labels,
            "invalid_labels": self.invalid_labels,
            "duplicate_hash_groups": self.duplicate_hash_groups,
            "leakage_groups": self.leakage_groups,
            "issues": self.issues,
            "warnings": self.warnings,
        }

    def summary_text(self) -> str:
        inv = self.class_to_idx
        lines = [
            f"YOLO11 detection dataset validation -- {self.workspace}",
            f"classes            : {len(inv)} "
            f"({', '.join(f'{k}={v}' for k, v in inv.items())})",
            f"images checked     : {self.images_checked}",
            "images per split   : "
            + " ".join(f"{s}={self.per_split_images[s]}" for s in VALID_SPLITS),
            "boxes per split    : "
            + " ".join(f"{s}={self.per_split_boxes[s]}" for s in VALID_SPLITS),
            "boxes per class    : "
            + " ".join(
                f"{name}={n}"
                for name, n in sorted(
                    self.per_class_boxes.items(),
                    key=lambda kv: self.class_to_idx.get(kv[0], 1 << 30),
                )
            ),
            f"missing labels     : {self.missing_labels} (issues)",
            f"empty labels       : {self.empty_labels} (warnings, unverified)",
            f"invalid labels     : {self.invalid_labels} (issues)",
            f"duplicate hashes   : {self.duplicate_hash_groups} groups",
            f"split leakage      : {self.leakage_groups} groups (issues)",
            f"issues             : {len(self.issues)}",
            f"warnings           : {len(self.warnings)}",
        ]
        lines += [f"[ISSUE] {m}" for m in self.issues]
        lines += [f"[WARN ] {m}" for m in self.warnings]
        lines.append(f"RESULT: {'VALID' if self.is_valid else 'NOT VALID'}")
        return "\n".join(lines)


def validate_detection_dataset(
    workspace: Path,
    class_to_idx: dict[str, int],
    manifest_path: Path | None = None,
    images_dir: Path | None = None,
    labels_dir: Path | None = None,
) -> DatasetReport:
    """Validate the YOLO detection dataset under ``workspace`` (read-only).

    Raises ``SetupError`` (with the exact missing paths and CVAT export
    instructions) when the workspace or any required directory/file is absent.
    """
    workspace = Path(workspace)
    manifest_path = Path(manifest_path) if manifest_path else workspace / "pilot_manifest.csv"
    images_dir = Path(images_dir) if images_dir else workspace / "images"
    labels_dir = Path(labels_dir) if labels_dir else workspace / "labels"

    missing: list[str] = []
    if not workspace.is_dir():
        missing.append(f"workspace directory: {workspace}")
    if not manifest_path.is_file():
        missing.append(f"manifest: {manifest_path}")
    if not images_dir.is_dir():
        missing.append(f"images directory: {images_dir}")
    if not labels_dir.is_dir():
        missing.append(f"labels directory: {labels_dir} (CVAT YOLO export not done yet)")
    if missing:
        raise SetupError(
            "dataset workspace is not ready; missing:\n  - "
            + "\n  - ".join(missing)
            + "\n\n"
            + CVAT_EXPORT_HINT
        )

    report = DatasetReport(workspace=workspace, class_to_idx=dict(class_to_idx))
    report.per_class_boxes = {name: 0 for name in class_to_idx}
    valid_class_ids = set(class_to_idx.values())
    name_of = {v: k for k, v in class_to_idx.items()}

    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SetupError(f"manifest is empty: {manifest_path}")
    for col in ("filename", "split"):
        if col not in rows[0]:
            raise SetupError(f"manifest missing required column {col!r}: {manifest_path}")

    seen_names: dict[str, str] = {}  # filename -> split
    hash_to_images: dict[str, list[tuple[str, str]]] = {}  # sha256 -> [(name, split)]

    for rec in rows:
        name = rec["filename"]
        split = (rec.get("split") or "").strip()
        stem = Path(name).stem
        report.images_checked += 1

        if split not in VALID_SPLITS:
            report.issues.append(
                f"{name}: split {split!r} not in {VALID_SPLITS} (manifest row)"
            )
            split_key = None
        else:
            split_key = split
            report.per_split_images[split] += 1

        if name in seen_names:
            report.issues.append(
                f"{name}: duplicate manifest row (splits {seen_names[name]!r} and {split!r})"
            )
        else:
            seen_names[name] = split

        img_path = images_dir / name
        if not img_path.is_file():
            report.issues.append(f"{name}: image file missing: {img_path}")
        else:
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
                    im.verify()
            except Exception as exc:  # noqa: BLE001 -- report any unreadable file
                report.issues.append(f"{name}: unreadable image ({exc})")
            else:
                for col, actual in (("width", width), ("height", height)):
                    want = (rec.get(col) or "").strip()
                    if want and int(want) != actual:
                        report.issues.append(
                            f"{name}: {col} {want} in manifest != {actual} on disk"
                        )
                digest = sha256_of(img_path)
                want_sha = (rec.get("sha256") or "").strip()
                if want_sha and want_sha != digest:
                    report.issues.append(
                        f"{name}: sha256 mismatch -- image changed since pilot build"
                    )
                hash_to_images.setdefault(digest, []).append((name, split))

        # NOTE: rec['class_name'] / rec['class_id'] are annotator priors only
        # (policy section 1). Boxes come exclusively from the label file below;
        # nothing is ever fabricated from image-level labels.
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.is_file():
            report.missing_labels += 1
            report.issues.append(
                f"{name}: missing label file {label_path.name} "
                "(pending annotation/export -- never inferred from class labels)"
            )
            continue
        parsed = parse_label_file(label_path, valid_class_ids)
        if parsed.problems:
            report.invalid_labels += 1
            for p in parsed.problems:
                report.issues.append(f"{label_path.name}: {p}")
        if parsed.n_boxes == 0 and not parsed.problems:
            report.empty_labels += 1
            report.warnings.append(
                f"{label_path.name}: empty label file (unverified_empty -- NOT a "
                "verified negative; record in review_log.csv, policy sections 3.5/9)"
            )
        if split_key is not None:
            report.per_split_boxes[split_key] += parsed.n_boxes
        for cid in parsed.class_ids:
            key = name_of.get(cid, str(cid))
            report.per_class_boxes[key] = report.per_class_boxes.get(key, 0) + 1

    # Orphan labels: files in labels/ with no manifest image.
    manifest_stems = {Path(r["filename"]).stem for r in rows}
    for p in sorted(labels_dir.glob("*.txt")):
        if p.stem not in manifest_stems:
            report.warnings.append(
                f"orphan label file (no manifest image): {p.name}"
            )

    # Duplicate image content and split leakage by sha256.
    for digest, members in sorted(hash_to_images.items()):
        if len(members) < 2:
            continue
        names = sorted(n for n, _ in members)
        splits = sorted({s for _, s in members})
        if len(splits) > 1:
            report.leakage_groups += 1
            report.issues.append(
                f"split leakage: identical image content in splits {splits}: "
                f"{', '.join(names)} (sha256 {digest[:12]}...)"
            )
        else:
            report.duplicate_hash_groups += 1
            report.warnings.append(
                f"duplicate image content within split {splits[0]!r}: "
                f"{', '.join(names)} (sha256 {digest[:12]}...)"
            )

    # Empty-class / empty-split warnings (report completeness, never fabricate).
    for cid in sorted(valid_class_ids):
        key = name_of.get(cid, str(cid))
        if report.per_class_boxes.get(key, 0) == 0:
            report.warnings.append(f"class {key!r} (id {cid}): zero boxes in dataset")
    for s in VALID_SPLITS:
        if report.per_split_images[s] == 0:
            report.warnings.append(f"split {s!r}: zero images")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument(
        "--json-report", type=Path, default=None,
        help="optional path to write a JSON summary of the validation",
    )
    args = parser.parse_args(argv)

    if not args.class_map.is_file():
        print(f"ERROR: class map not found: {args.class_map}", file=sys.stderr)
        return 2
    with args.class_map.open(encoding="utf-8") as f:
        class_to_idx = json.load(f)

    try:
        report = validate_detection_dataset(
            args.workspace,
            class_to_idx,
            manifest_path=args.manifest,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
        )
    except SetupError as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return 2

    print(report.summary_text())
    if not report.is_valid:
        print(
            "\nFix every [ISSUE] line in CVAT (or the workspace) and re-export; "
            "never hand-edit the .txt label files."
        )

    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"json report written: {args.json_report}")

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

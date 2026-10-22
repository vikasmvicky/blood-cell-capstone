"""Stage 1 tests for src/detection/dataset.py (YOLO11 detection validation).

Fixtures are tiny synthetic workspaces (solid-color PNGs + hand-written YOLO
label lines) under tmp_path. They exercise the validator's code paths only --
no real pilot label is created and no box is ever fabricated from the
image-level class columns of the manifest.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from src.detection.dataset import (
    SetupError,
    main,
    sha256_of,
    validate_detection_dataset,
)

# Real project class mapping (outputs/reports/class_to_idx.json).
CLASS_TO_IDX = {
    "basophil": 0,
    "eosinophil": 1,
    "erythroblast": 2,
    "ig": 3,
    "lymphocyte": 4,
    "monocyte": 5,
    "neutrophil": 6,
    "platelet": 7,
}

# filename, split, class_name, color -- 4 distinct images across the 3 splits.
SPECS = [
    ("BA_1.jpg", "train", "basophil", (200, 30, 30)),
    ("BNE_1.jpg", "train", "neutrophil", (30, 200, 30)),
    ("EO_1.jpg", "val", "eosinophil", (30, 30, 200)),
    ("PLT_1.jpg", "test", "platelet", (200, 200, 30)),
]

DEFAULT_LABELS = {
    "BA_1": "0 0.5 0.5 0.4 0.4\n",
    "BNE_1": "6 0.5 0.5 0.4 0.4\n",
    "EO_1": "1 0.5 0.5 0.3 0.3\n",
    "PLT_1": "7 0.5 0.5 0.2 0.2\n",
}


def make_workspace(
    tmp_path: Path,
    labels: dict[str, str] | None = None,
    skip_labels: tuple[str, ...] = (),
    specs=SPECS,
) -> tuple[Path, Path]:
    """Build a synthetic pilot-style workspace; returns (workspace, class_map)."""
    labels = {**DEFAULT_LABELS, **(labels or {})}
    ws = tmp_path / "yolo_ws"
    images = ws / "images"
    labels_dir = ws / "labels"
    images.mkdir(parents=True)
    labels_dir.mkdir()

    rows = []
    for i, (name, split, cls, color) in enumerate(specs, start=1):
        img = images / name
        Image.new("RGB", (32, 32), color).save(img)
        stem = Path(name).stem
        rows.append(
            {
                "pilot_index": i,
                "class_name": cls,
                "class_id": CLASS_TO_IDX[cls],
                "split": split,
                "filename": name,
                "source_rel_path": f"{cls}/{name}",
                "workspace_image_path": f"data/x/images/{name}",
                "width": 32,
                "height": 32,
                "sha256": sha256_of(img),
            }
        )
        if stem not in skip_labels:
            (labels_dir / f"{stem}.txt").write_text(labels[stem], encoding="utf-8")

    manifest = ws / "pilot_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps(CLASS_TO_IDX), encoding="utf-8")
    return ws, class_map


def test_valid_dataset_passes_and_counts(tmp_path):
    ws, _ = make_workspace(tmp_path)
    report = validate_detection_dataset(ws, CLASS_TO_IDX)
    assert report.is_valid, report.issues
    assert report.images_checked == 4
    assert report.per_split_images == {"train": 2, "val": 1, "test": 1}
    assert report.per_split_boxes == {"train": 2, "val": 1, "test": 1}
    assert report.per_class_boxes["basophil"] == 1
    assert report.per_class_boxes["neutrophil"] == 1
    assert report.missing_labels == 0
    assert report.empty_labels == 0
    assert report.invalid_labels == 0
    assert report.leakage_groups == 0
    assert "RESULT: VALID" in report.summary_text()


def test_missing_label_is_issue_and_never_fabricated(tmp_path):
    ws, cmap = make_workspace(tmp_path, skip_labels=("EO_1",))
    report = validate_detection_dataset(ws, CLASS_TO_IDX)
    assert not report.is_valid
    assert report.missing_labels == 1
    assert any("missing label file EO_1.txt" in m for m in report.issues)
    # Nothing was invented for the un-annotated image.
    assert not (ws / "labels" / "EO_1.txt").exists()
    # And the image-level class prior (eosinophil) did not become a box.
    assert report.per_class_boxes["eosinophil"] == 0
    assert main(["--workspace", str(ws), "--class-map", str(cmap)]) == 1


def test_invalid_label_lines_detected(tmp_path):
    bad = "9 0.5 0.5 0.4 0.4\n0 1.5 0.5 0.4 0.4\n0 0.5 0.5\n"
    ws, _ = make_workspace(tmp_path, labels={"BA_1": bad})
    report = validate_detection_dataset(ws, CLASS_TO_IDX)
    assert not report.is_valid
    assert report.invalid_labels == 1
    joined = "\n".join(report.issues)
    assert "class_id 9" in joined
    assert "x_center=1.5" in joined
    assert "expected 5 fields" in joined
    # Invalid lines are never counted as boxes.
    assert report.per_class_boxes["basophil"] == 0


def test_box_outside_image_detected(tmp_path):
    ws, _ = make_workspace(tmp_path, labels={"BA_1": "0 0.95 0.5 0.2 0.2\n"})
    report = validate_detection_dataset(ws, CLASS_TO_IDX)
    assert not report.is_valid
    assert any("outside image horizontally" in m for m in report.issues)


def test_cross_split_duplicate_hash_is_leakage(tmp_path):
    ws, _ = make_workspace(tmp_path)
    # Copy train image bytes into a new val image (different name, same content).
    name = "BA_1_copy.jpg"
    src = ws / "images" / "BA_1.jpg"
    dst = ws / "images" / name
    dst.write_bytes(src.read_bytes())
    (ws / "labels" / "BA_1_copy.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    with (ws / "pilot_manifest.csv").open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=[
            "pilot_index", "class_name", "class_id", "split", "filename",
            "source_rel_path", "workspace_image_path", "width", "height", "sha256",
        ]).writerow({
            "pilot_index": 5, "class_name": "basophil", "class_id": 0,
            "split": "val", "filename": name, "source_rel_path": f"basophil/{name}",
            "workspace_image_path": f"data/x/images/{name}", "width": 32,
            "height": 32, "sha256": sha256_of(dst),
        })
    report = validate_detection_dataset(ws, CLASS_TO_IDX)
    assert not report.is_valid
    assert report.leakage_groups == 1
    assert any("split leakage" in m and "BA_1" in m for m in report.issues)


def test_empty_label_is_warning_not_issue(tmp_path):
    ws, cmap = make_workspace(tmp_path, labels={"PLT_1": ""})
    report = validate_detection_dataset(ws, CLASS_TO_IDX)
    assert report.is_valid, report.issues
    assert report.empty_labels == 1
    assert any("unverified_empty" in m for m in report.warnings)
    assert main(["--workspace", str(ws), "--class-map", str(cmap)]) == 0


def test_missing_labels_dir_is_setup_error_with_cvat_instructions(tmp_path):
    ws, cmap = make_workspace(tmp_path)
    for p in (ws / "labels").glob("*.txt"):
        p.unlink()
    (ws / "labels").rmdir()
    with pytest.raises(SetupError) as exc:
        validate_detection_dataset(ws, CLASS_TO_IDX)
    msg = str(exc.value)
    assert str(ws / "labels") in msg
    assert "CVAT" in msg and "YOLO 1.1" in msg
    assert main(["--workspace", str(ws), "--class-map", str(cmap)]) == 2


def test_modified_image_flagged_by_sha256(tmp_path):
    ws, _ = make_workspace(tmp_path)
    Image.new("RGB", (32, 32), (1, 2, 3)).save(ws / "images" / "BA_1.jpg")
    report = validate_detection_dataset(ws, CLASS_TO_IDX)
    assert not report.is_valid
    assert any("BA_1.jpg: sha256 mismatch" in m for m in report.issues)

"""Objective traceability data for the blood-cell capstone project.

The capstone presentation defines six project objectives. This module records,
for each one, the honest current status inside *this* repository, the source
files that implement it, the on-disk evidence, the verification that backs the
status, and the remaining gaps. It is the single source of truth used by the
Research Overview page and by ``tests/test_objectives.py``; the readable
version lives in ``docs/OBJECTIVE_TRACEABILITY.md``.

Deliberately pure standard library (no Streamlit, no third-party imports) so
tests, scripts and CI tooling can import it anywhere.

Scope notes that constrain every status below:

* The deployed and evaluated model in this repository is a **fine-tuned
  MobileNetV2 image classifier** (whole-image, 8 PBC classes). It is not an
  object detector and not a segmentation model.
* The capstone presentation's YOLO objectives (1 and 2) refer to a detection
  pipeline that does not exist in this repository; they are recorded as
  ``not_implemented`` rather than being papered over.
* The 2025 Scientific Reports paper is a **methodological reference only**.

Reference paper (methodology only - none of its results are claimed here):
Scientific Reports (2025), "Efficient blood cell classification from
microscopic smear images using U-Net segmentation and a lightweight CNN",
DOI: 10.1038/s41598-025-26947-5. Its pipeline (U-Net segmentation,
watershed separation, lightweight CNN, nine categories including erythrocyte,
its own dataset) is *not* implemented in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATUS_IMPLEMENTED = "implemented"
STATUS_PARTIAL = "partial"
STATUS_NOT_IMPLEMENTED = "not_implemented"

VALID_STATUSES = (STATUS_IMPLEMENTED, STATUS_PARTIAL, STATUS_NOT_IMPLEMENTED)

STATUS_LABELS = {
    STATUS_IMPLEMENTED: "Implemented",
    STATUS_PARTIAL: "Partial",
    STATUS_NOT_IMPLEMENTED: "Not implemented",
}

PAPER_DOI = "10.1038/s41598-025-26947-5"

PAPER_REFERENCE = (
    "Scientific Reports (2025), 'Efficient blood cell classification from microscopic smear "
    "images using U-Net segmentation and a lightweight CNN', DOI: "
    f"{PAPER_DOI}. Methodological reference only: its U-Net + watershed + nine-class "
    "pipeline (including erythrocyte) and its reported metrics belong to its own dataset and "
    "experimental setup; they are not results of this project and are not reproduced here."
)


@dataclass(frozen=True)
class Objective:
    """One capstone objective and its honest traceability status."""

    number: int
    title: str
    status: str
    summary: str
    implemented_in: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    verification: str = ""
    gaps: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


OBJECTIVES: tuple[Objective, ...] = (
    Objective(
        number=1,
        title="Reproduce and benchmark the existing YOLOv8 baseline",
        status=STATUS_NOT_IMPLEMENTED,
        summary=(
            "This repository contains no YOLOv8 code, no `ultralytics` dependency, no detector "
            "checkpoint and no detection benchmark. The model that exists here is a whole-image "
            "MobileNetV2 classifier. In addition, the PBC dataset used by this project provides "
            "image-level class labels only - no bounding boxes or masks - so a YOLOv8 detection "
            "baseline cannot be reproduced from the available data alone."
        ),
        implemented_in=(),
        evidence=(),
        verification=(
            "Verified by repository inspection: no YOLO modules under `src/`, no `ultralytics` "
            "entry in `requirements.txt`, no detection artifacts under `outputs/`. "
            "`configs/config.json` declares architecture `mobilenet_v2`."
        ),
        gaps=(
            "Add an object-detection framework (e.g. ultralytics) as a constrained dependency.",
            "Obtain or annotate detection ground truth (bounding boxes/masks) - the PBC images "
            "have class labels only.",
            "Train YOLOv8 on a fixed, leakage-safe split and benchmark against the classifier "
            "with clearly separated metrics (detection mAP vs classification accuracy).",
        ),
        limitations=(
            "Marking this objective complete without detection ground truth would fabricate a "
            "capability; it is recorded as not implemented instead.",
        ),
    ),
    Objective(
        number=2,
        title="Upgrade to a newer YOLO architecture (YOLOv9 / YOLO11) if supported",
        status=STATUS_NOT_IMPLEMENTED,
        summary=(
            "Blocked by objective 1: there is no YOLOv8 baseline in this repository to upgrade "
            "from, and no detection data to train any YOLO variant on. No YOLO dependency was "
            "added, because an untrained detector would be an unusable showcase, not an upgrade."
        ),
        implemented_in=(),
        evidence=(),
        verification=(
            "Same inspection as objective 1. The delivered classifier remains the only model "
            "with trained weights (`outputs/models/best_mobilenet_finetuned.pt`)."
        ),
        gaps=(
            "Complete objective 1 first; then compare YOLOv9/YOLO11 variants under matched data, "
            "splits and compute, reporting both accuracy and latency.",
        ),
        limitations=(
            "CPU-only development environment; training modern YOLO models would need GPU "
            "resources and re-annotation review.",
        ),
    ),
    Objective(
        number=3,
        title="Handle class imbalance with weighting, augmentation and/or oversampling",
        status=STATUS_IMPLEMENTED,
        summary=(
            "The delivered checkpoint was trained with inverse-frequency class-weighted "
            "cross-entropy computed from the training-split counts. A WeightedRandomSampler "
            "oversampling variant also exists as a documented experiment script. Training adds "
            "mild augmentation (random flips, <=15 degree rotations), the split is stratified "
            "by duplicate group, and per-class precision/recall/F1 are measured and displayed "
            "so imbalance effects stay visible."
        ),
        implemented_in=(
            "src/models/train_mobilenet_finetune.py",
            "src/models/train_mobilenet_sampler.py",
            "src/data/image_dataset.py",
            "src/data/split_dataset.py",
        ),
        evidence=(
            "outputs/reports/mobilenet_finetuned_training_history.csv",
            "outputs/reports/mobilenet_finetuned_test_metrics.json",
            "outputs/reports/split_integrity.json",
        ),
        verification=(
            "Per-class metrics on the 2,562-image held-out test split "
            "(`mobilenet_finetuned_test_metrics.json`) show minority classes remain strong "
            "(e.g. basophil recall 0.967, platelet F1 0.993) while exposing honest weaknesses "
            "(basophil precision 0.843, IG recall 0.834); the Evaluation Dashboard renders "
            "these numbers per class."
        ),
        gaps=(
            "The sampler-variant run is not archived in outputs/reports (no "
            "`mobilenet_sampler_history.csv`), so the oversampling path is implemented in code "
            "but has no recorded result in this archive.",
        ),
        limitations=(
            "Residual per-class variance (IG, basophil precision) shows weighting mitigates "
            "but does not eliminate imbalance effects.",
        ),
    ),
    Objective(
        number=4,
        title="Improve WBC and platelet performance via resolution/model-config experiments",
        status=STATUS_PARTIAL,
        summary=(
            "WBC and platelet performance is *measured* rigorously (per-class test metrics, "
            "top confusions, robustness degradations, Grad-CAM faithfulness), and one "
            "controlled experiment exists: a classical watershed centre-crop study testing "
            "whether pre-cropping the cell region changes classification accuracy. However, no "
            "input-resolution sweep or alternative model-configuration training runs are "
            "archived; the deployed model is a single 224x224 MobileNetV2 configuration."
        ),
        implemented_in=(
            "src/experiments/segmentation_study.py",
            "src/segmentation/watershed.py",
            "src/evaluation/robustness.py",
        ),
        evidence=(
            "outputs/reports/segmentation_study.json",
            "outputs/reports/robustness.json",
            "outputs/reports/mobilenet_finetuned_test_metrics.json",
            "outputs/reports/benchmark.json",
        ),
        verification=(
            "Held-out per-class metrics quantify each WBC/platelet class; "
            "`segmentation_study.json` records that watershed centre-cropping did not improve "
            "accuracy (reported as measured, including the null result)."
        ),
        gaps=(
            "Controlled resolution experiments (e.g. 160/224/320 px) with matched training "
            "budgets and validation-based selection.",
            "Alternative lightweight architectures trained and compared on the same split.",
            "Targeted error analysis of the IG <-> basophil <-> monocyte confusions to guide "
            "configuration changes.",
        ),
        limitations=(
            "The reference paper's U-Net segmentation improvements cannot be claimed here: "
            "this dataset ships no ground-truth masks, and the classical watershed experiment "
            "did not yield an accuracy gain.",
        ),
    ),
    Objective(
        number=5,
        title="Web interface for image upload and annotated results",
        status=STATUS_IMPLEMENTED,
        summary=(
            "A Streamlit research application (`app.py` + `app_ui/`) supports PNG/JPG/JPEG "
            "upload with validation (type, size, decodability, dimensions), prediction with "
            "top-k probabilities and uncertainty indicators, Grad-CAM overlay as the "
            "annotation, heuristic image-quality warnings, evaluation/calibration dashboards "
            "and downloadable reports. It runs on CPU and degrades gracefully when the "
            "checkpoint or reports are missing."
        ),
        implemented_in=(
            "app.py",
            "app_ui/prediction.py",
            "app_ui/explainability.py",
            "app_ui/evaluation_page.py",
            "app_ui/calibration_page.py",
            "app_ui/overview.py",
            "app_ui/methodology.py",
            "src/inference/preprocess.py",
            "src/inference/predictor.py",
            "src/inference/quality.py",
            "src/explainability/gradcam.py",
        ),
        evidence=(
            "tests/test_app_smoke.py",
            "outputs/logs/streamlit_smoke.log",
        ),
        verification=(
            "`tests/test_app_smoke.py` renders every section via Streamlit's AppTest harness "
            "with valid, missing and corrupt checkpoints (73-test suite passing)."
        ),
        gaps=(),
        limitations=(
            "'Annotated results' are Grad-CAM heat-map overlays, not bounding boxes; the app "
            "explicitly states it is a research prototype, not a medical device.",
        ),
    ),
    Objective(
        number=6,
        title="Detection and a downloadable summary report",
        status=STATUS_PARTIAL,
        summary=(
            "The downloadable summary report is implemented: per-image prediction exports "
            "(JSON/CSV with probabilities, uncertainty and warnings) plus dashboard downloads "
            "of the full evaluation, calibration and robustness reports. Detection "
            "(localisation, bounding boxes, cell counting) is *not* implemented - the model is "
            "a whole-image classifier, and drawing boxes or counting cells would fabricate a "
            "capability the model does not have."
        ),
        implemented_in=(
            "app_ui/prediction.py",
            "app_ui/evaluation_page.py",
            "app_ui/calibration_page.py",
        ),
        evidence=(
            "outputs/reports/mobilenet_finetuned_test_metrics.json",
            "outputs/reports/test_predictions.csv",
            "outputs/reports/top_confusions.csv",
        ),
        verification=(
            "Download buttons serve the exact on-disk report bytes with correct MIME types "
            "(covered by the app smoke tests); prediction export rows are generated from the "
            "live predictor output."
        ),
        gaps=(
            "Real detection/localisation depends on objectives 1-2 (annotated detection data "
            "and a trained detector).",
        ),
        limitations=(
            "No cell counting or localisation is offered; per-image summaries cover one cell "
            "crop per upload, matching the training distribution.",
        ),
    ),
)

assert all(o.status in VALID_STATUSES for o in OBJECTIVES)
assert [o.number for o in OBJECTIVES] == [1, 2, 3, 4, 5, 6]


def get_objective(number: int) -> Objective:
    """Return the objective with the given 1-based number."""
    for objective in OBJECTIVES:
        if objective.number == number:
            return objective
    raise KeyError(f"No objective numbered {number}")


def status_counts() -> dict[str, int]:
    """Return {'implemented': n, 'partial': n, 'not_implemented': n}."""
    counts = {status: 0 for status in VALID_STATUSES}
    for objective in OBJECTIVES:
        counts[objective.status] += 1
    return counts


def missing_evidence(root: Path | None = None) -> list[tuple[int, str]]:
    """Evidence files claimed by implemented/partial objectives but absent on disk.

    ``not_implemented`` objectives claim no evidence by construction, so they
    never contribute here. Returns ``(objective_number, relative_path)`` pairs.
    """
    base = Path(root) if root is not None else PROJECT_ROOT
    missing: list[tuple[int, str]] = []
    for objective in OBJECTIVES:
        if objective.status == STATUS_NOT_IMPLEMENTED:
            continue
        for rel in (*objective.implemented_in, *objective.evidence):
            if not (base / rel).exists():
                missing.append((objective.number, rel))
    return missing

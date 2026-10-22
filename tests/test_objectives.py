"""Tests for the objective traceability matrix (`app_ui/objectives.py`).

These tests pin the *honest* status of each capstone objective so the UI and
documentation cannot silently upgrade a status without evidence:
- exactly six objectives, numbered 1-6;
- the YOLO objectives (1, 2) stay "not implemented" and claim no files/evidence;
- every evidence/implementation path claimed by an implemented or partial
  objective actually exists on disk;
- the module stays pure standard library (no Streamlit import);
- the readable doc matches the pinned statuses and cites the paper correctly.
"""

from __future__ import annotations

import ast
from pathlib import Path

from app_ui import objectives
from app_ui.objectives import (
    OBJECTIVES,
    STATUS_IMPLEMENTED,
    STATUS_NOT_IMPLEMENTED,
    STATUS_PARTIAL,
    STATUS_LABELS,
    VALID_STATUSES,
    PROJECT_ROOT,
)

DOC = PROJECT_ROOT / "docs" / "OBJECTIVE_TRACEABILITY.md"

# The honest status of every objective, pinned. Updating one of these requires
# real new evidence plus a matching doc update - that is the point.
PINNED_STATUS = {
    1: STATUS_NOT_IMPLEMENTED,  # YOLOv8 baseline: no code, no detection data
    2: STATUS_NOT_IMPLEMENTED,  # YOLOv9/11 upgrade: blocked by 1
    3: STATUS_IMPLEMENTED,      # class-weighted CE used for the delivered checkpoint
    4: STATUS_PARTIAL,          # measured per-class + one study; no resolution sweep
    5: STATUS_IMPLEMENTED,      # Streamlit app with upload, Grad-CAM, downloads
    6: STATUS_PARTIAL,          # summary downloads yes; detection no
}


def test_exactly_six_objectives_numbered_1_to_6():
    assert len(OBJECTIVES) == 6
    assert [o.number for o in OBJECTIVES] == [1, 2, 3, 4, 5, 6]


def test_statuses_are_valid_and_pinned():
    for objective in OBJECTIVES:
        assert objective.status in VALID_STATUSES, objective.number
        assert objective.status == PINNED_STATUS[objective.number], (
            f"objective {objective.number} status changed - requires real evidence "
            "and a docs/OBJECTIVE_TRACEABILITY.md update"
        )


def test_all_objectives_have_titles_and_summaries():
    for objective in OBJECTIVES:
        assert objective.title.strip(), objective.number
        assert len(objective.summary) > 80, f"objective {objective.number}: summary too thin"


def test_yolo_objectives_claim_no_files_or_evidence():
    """Objectives 1 and 2 must not pretend any implementation exists."""
    for number in (1, 2):
        objective = objectives.get_objective(number)
        assert objective.implemented_in == ()
        assert objective.evidence == ()
        assert objective.gaps, "an unimplemented objective must document its gaps"


def test_implemented_and_partial_objectives_reference_existing_files():
    """Every claimed path must exist on disk (no aspirational evidence)."""
    missing = objectives.missing_evidence()
    assert missing == [], f"objectives claim missing files: {missing}"


def test_status_counts_are_consistent():
    counts = objectives.status_counts()
    assert sum(counts.values()) == 6
    assert counts[STATUS_IMPLEMENTED] == 2
    assert counts[STATUS_PARTIAL] == 2
    assert counts[STATUS_NOT_IMPLEMENTED] == 2


def test_get_objective_lookup_and_error():
    assert objectives.get_objective(5).status == STATUS_IMPLEMENTED
    try:
        objectives.get_objective(99)
    except KeyError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("expected KeyError for unknown objective number")


def test_module_imports_only_standard_library():
    """The objectives module must stay usable without Streamlit or third-party deps."""
    source = (PROJECT_ROOT / "app_ui" / "objectives.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "streamlit" not in imported
    assert imported <= {"__future__", "dataclasses", "pathlib"}, imported


def test_traceability_doc_matches_pinned_statuses_and_cites_paper():
    assert DOC.is_file(), "docs/OBJECTIVE_TRACEABILITY.md is missing"
    text = DOC.read_text(encoding="utf-8")
    for objective in OBJECTIVES:
        # The doc must mention every objective and its pinned status label.
        key_phrase = objective.title.split("(")[0].strip()
        assert key_phrase[:40] in text, f"doc does not mention objective {objective.number}"
    # Status labels present for each pinned status.
    assert "Not implemented" in text
    assert "Implemented" in text
    assert "Partial" in text
    # The 2025 paper is cited as a methodological reference only, with its DOI,
    # and is not claimed as this project's result.
    assert objectives.PAPER_DOI in text
    assert "methodological reference only" in text.lower()
    assert "erythrocyte" in text  # scope distinction from the paper's nine classes
    # The doc must not advertise the historical, unreproduced 97.07% as current.
    assert "97.07" not in text

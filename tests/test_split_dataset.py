
import json

from pathlib import Path

import pandas as pd
import pytest

import src.data.split_dataset as split_module

from src.data.split_dataset import (
    assign_splits,
    find_label_conflicts,
    prepare_split_data,
    save_split_outputs,
    verify_no_leakage,
)


def make_sample_data():
    """Create a small dataset for testing split logic."""

    return pd.DataFrame(
        {
            "rel_path": [
                "basophil/a.jpg",
                "basophil/b.jpg",
                "eosinophil/c.jpg",
                "neutrophil/d.jpg",
            ],
            "class": [
                "basophil",
                "basophil",
                "eosinophil",
                "neutrophil",
            ],
            "readable": [True, True, True, True],
            "sha256": [
                "hash_a",
                "hash_a",
                "hash_b",
                "hash_b",
            ],
        }
    )


def test_find_label_conflicts():
    df = make_sample_data()

    conflicts = find_label_conflicts(df)

    assert conflicts == {"hash_b"}


def test_prepare_split_data_excludes_conflicting_images():
    df = make_sample_data()

    usable_df, excluded_df = prepare_split_data(df)

    assert len(usable_df) == 2
    assert len(excluded_df) == 2
    assert set(excluded_df["exclusion_reason"]) == {
        "cross_class_duplicate_conflict"
    }


def test_assign_splits_is_reproducible():
    df = make_sample_data()
    usable_df, _ = prepare_split_data(df)

    first = assign_splits(usable_df, seed=42)
    second = assign_splits(usable_df, seed=42)

    assert first["split"].tolist() == second["split"].tolist()


def test_verify_no_leakage_rejects_cross_split_hashes():
    df = pd.DataFrame(
        {
            "sha256": ["same_hash", "same_hash"],
            "duplicate_group": [1, 1],
            "split": ["train", "test"],
        }
    )

    with pytest.raises(ValueError, match="leakage detected"):
        verify_no_leakage(df)


def test_assign_splits_rejects_invalid_ratios():
    df = make_sample_data()
    usable_df, _ = prepare_split_data(df)

    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_splits(
            usable_df,
            train_ratio=0.70,
            val_ratio=0.20,
            test_ratio=0.20,
        )


def test_prepare_split_data_excludes_unreadable_images():
    df = make_sample_data()
    df.loc[0, "readable"] = False

    usable_df, excluded_df = prepare_split_data(df)

    assert len(excluded_df) == 3
    assert "unreadable" in set(excluded_df["exclusion_reason"])
    assert "cross_class_duplicate_conflict" in set(
        excluded_df["exclusion_reason"]
    )


def test_save_split_outputs_records_metadata(tmp_path, monkeypatch):
    """Verify the selected seed and ratios are saved."""

    sample = make_sample_data()
    usable_df, excluded_df = prepare_split_data(sample)

    split_df = assign_splits(
        usable_df,
        seed=123,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    monkeypatch.setattr(
        split_module,
        "PROJECT_ROOT",
        tmp_path,
    )

    save_split_outputs(
        split_df,
        excluded_df,
        seed=123,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    report_dir = tmp_path / "outputs" / "reports"

    with (report_dir / "split_summary.json").open(
        encoding="utf-8"
    ) as file:
        summary = json.load(file)

    assert summary["random_seed"] == 123
    assert summary["split_ratios"] == {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
    }
    assert set(summary["class_to_idx"]) == {"basophil"}

    markdown = (report_dir / "split_report.md").read_text(
        encoding="utf-8"
    )

    assert "Random seed: 123" in markdown
    assert "70% train" in markdown


"""Create reproducible train, validation, and test splits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_NAMES = ("train", "val", "test")

REQUIRED_COLUMNS = {
    "rel_path",
    "class",
    "readable",
    "sha256",
}


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load and validate the Phase 1 image manifest."""

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    df = pd.read_csv(
        manifest_path,
        dtype={"sha256": "string"},
    )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df["readable"] = (
        df["readable"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

    if df["readable"].isna().any():
        raise ValueError(
            "The readable column contains invalid values."
        )

    return df


def find_label_conflicts(df: pd.DataFrame) -> set[str]:
    """Find hashes assigned to more than one class."""

    usable = df[df["readable"]].copy()

    valid_hash = (
        usable["sha256"].notna()
        & usable["sha256"].astype(str).str.strip().ne("")
    )
    usable = usable[valid_hash]

    class_counts = usable.groupby("sha256")["class"].nunique()

    return set(
        class_counts[class_counts > 1].index.tolist()
    )


def group_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Assign identical images to the same duplicate group."""

    result = df.copy()

    # Missing hashes are assigned unique groups. They must not
    # be treated as identical merely because both hashes are missing.
    hash_values = result["sha256"].astype("string").copy()

    missing_hash = (
        hash_values.isna()
        | hash_values.str.strip().eq("").fillna(True)
    )

    hash_values.loc[missing_hash] = (
        "missing:"
        + result.loc[missing_hash, "rel_path"].astype(str)
    )

    result["duplicate_group"] = (
        hash_values.groupby(hash_values, sort=False)
        .ngroup()
    )

    return result


def prepare_split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate usable images from excluded images."""

    result = group_exact_duplicates(df)

    conflicting_hashes = find_label_conflicts(result)

    unreadable_mask = ~result["readable"]

    missing_hash_mask = (
        result["sha256"].isna()
        | result["sha256"].astype("string").str.strip().eq("").fillna(True)
    )

    conflict_mask = result["sha256"].isin(conflicting_hashes)

    # Exclude unreadable images, missing hashes, and cross-class
    # conflicts so that the split process is leakage-aware.
    excluded_mask = (
        unreadable_mask
        | missing_hash_mask
        | conflict_mask
    )

    excluded_df = result.loc[excluded_mask].copy()
    usable_df = result.loc[~excluded_mask].copy()

    excluded_df["exclusion_reason"] = ""

    excluded_df.loc[
        unreadable_mask.loc[excluded_df.index],
        "exclusion_reason",
    ] = "unreadable"

    excluded_df.loc[
        missing_hash_mask.loc[excluded_df.index]
        & ~unreadable_mask.loc[excluded_df.index],
        "exclusion_reason",
    ] = "missing_sha256"

    excluded_df.loc[
        conflict_mask.loc[excluded_df.index],
        "exclusion_reason",
    ] = "cross_class_duplicate_conflict"

    if usable_df.empty:
        raise ValueError(
            "No usable images remain after exclusions."
        )

    return usable_df, excluded_df


def assign_splits(
    usable_df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Assign complete duplicate groups to train, val, or test."""

    ratios = [train_ratio, val_ratio, test_ratio]

    if any(not np.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be positive.")

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    if usable_df.empty:
        raise ValueError("Cannot split an empty dataset.")

    required = {"class", "duplicate_group", "sha256"}

    missing = required - set(usable_df.columns)

    if missing:
        raise ValueError(
            f"Split data is missing columns: {sorted(missing)}"
        )

    if usable_df["duplicate_group"].isna().any():
        raise ValueError("Duplicate groups cannot be missing.")

    rng = np.random.default_rng(seed)

    result = usable_df.copy()
    result["split"] = ""

    for class_name in sorted(result["class"].unique()):
        class_mask = result["class"] == class_name
        class_df = result.loc[class_mask]

        # Each duplicate group is an indivisible assignment unit.
        groups = (
            class_df.groupby("duplicate_group")
            .size()
            .rename("image_count")
            .reset_index()
        )

        groups = groups.iloc[
            rng.permutation(len(groups))
        ].reset_index(drop=True)

        total_images = int(groups["image_count"].sum())

        targets = {
            "train": total_images * train_ratio,
            "val": total_images * val_ratio,
            "test": total_images * test_ratio,
        }

        assigned_counts = {
            name: 0 for name in SPLIT_NAMES
        }
        assigned_groups = {}

        for _, group in groups.iterrows():
            group_id = group["duplicate_group"]
            image_count = int(group["image_count"])

            # Choose the split with the largest remaining target.
            # Stable ordering makes ties deterministic for a given
            # sequence of shuffled groups.
            deficits = {
                name: targets[name] - assigned_counts[name]
                for name in SPLIT_NAMES
            }

            split_name = max(
                SPLIT_NAMES,
                key=lambda name: deficits[name],
            )

            assigned_groups[group_id] = split_name
            assigned_counts[split_name] += image_count

        result.loc[class_mask, "split"] = (
            result.loc[class_mask, "duplicate_group"]
            .map(assigned_groups)
        )

    return result


def verify_no_leakage(split_df: pd.DataFrame) -> None:
    """Verify hashes and duplicate groups do not cross splits."""

    if split_df.empty:
        raise ValueError("Cannot verify an empty split dataset.")

    if split_df["split"].isna().any():
        raise ValueError("Some images have no assigned split.")

    if not split_df["split"].isin(SPLIT_NAMES).all():
        raise ValueError("An invalid split name was found.")

    hash_split_counts = (
        split_df.groupby("sha256")["split"].nunique()
    )

    if (hash_split_counts > 1).any():
        raise ValueError(
            "Data leakage detected: an exact hash appears "
            "in multiple splits."
        )

    group_split_counts = (
        split_df.groupby("duplicate_group")["split"].nunique()
    )

    if (group_split_counts > 1).any():
        raise ValueError(
            "Data leakage detected: a duplicate group "
            "appears in multiple splits."
        )

    print("Leakage check passed!")


def save_split_outputs(
    split_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """Save split assignments, mappings, and reports."""

    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    split_df.to_csv(
        report_dir / "splits.csv",
        index=False,
    )

    excluded_df.to_csv(
        report_dir / "excluded.csv",
        index=False,
    )

    classes = sorted(split_df["class"].unique())

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    with (report_dir / "class_to_idx.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(class_to_idx, file, indent=4)

    split_counts = (
        split_df["split"]
        .value_counts()
        .reindex(SPLIT_NAMES, fill_value=0)
    )

    summary = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "random_seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "total_usable_images": int(len(split_df)),
        "total_excluded_images": int(len(excluded_df)),
        "split_counts": {
            name: int(split_counts[name])
            for name in SPLIT_NAMES
        },
        "split_percentages": {
            name: round(
                float(split_counts[name] / len(split_df) * 100),
                2,
            )
            for name in SPLIT_NAMES
        },
        "class_to_idx": class_to_idx,
    }

    with (report_dir / "split_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    save_markdown_report(
        split_df,
        excluded_df,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    print("\nSaved split outputs to:", report_dir)
    print("Files created:")
    print("- splits.csv")
    print("- excluded.csv")
    print("- class_to_idx.json")
    print("- split_summary.json")
    print("- split_report.md")


def save_markdown_report(
    split_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """Write a human-readable split report."""

    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "split_report.md"

    split_counts = (
        split_df["split"]
        .value_counts()
        .reindex(SPLIT_NAMES, fill_value=0)
    )

    class_distribution = pd.crosstab(
        split_df["class"],
        split_df["split"],
    ).reindex(columns=SPLIT_NAMES, fill_value=0)

    lines = [
        "# Blood Cell Dataset Split Report",
        "",
        "## Dataset summary",
        "",
        f"- Usable images: {len(split_df)}",
        f"- Excluded images: {len(excluded_df)}",
        (
            f"- Split ratios: {train_ratio:.0%} train, "
            f"{val_ratio:.0%} validation, "
            f"{test_ratio:.0%} test"
        ),
        f"- Random seed: {seed}",
        "",
        "## Split distribution",
        "",
        "| Split | Images | Percentage |",
        "|---|---:|---:|",
    ]

    for split_name in SPLIT_NAMES:
        count = int(split_counts[split_name])
        percentage = count / len(split_df) * 100

        lines.append(
            f"| {split_name} | {count} | {percentage:.2f}% |"
        )

    lines.extend([
        "",
        "## Class distribution",
        "",
        "| Class | Train | Validation | Test |",
        "|---|---:|---:|---:|",
    ])

    for class_name, row in class_distribution.iterrows():
        lines.append(
            f"| {class_name} | {int(row['train'])} "
            f"| {int(row['val'])} | {int(row['test'])} |"
        )

    lines.extend([
        "",
        "## Exclusions",
        "",
    ])

    if excluded_df.empty:
        lines.append("No images were excluded.")
    else:
        reasons = excluded_df["exclusion_reason"].value_counts()

        for reason, count in reasons.items():
            lines.append(
                f"- {reason}: {int(count)} image(s)"
            )

    lines.extend([
        "",
        "## Leakage verification",
        "",
        "The exact-hash and duplicate-group leakage checks passed.",
        "",
    ])

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Markdown report saved to:", report_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Create leakage-aware dataset splits."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "reports"
            / "image_manifest.csv"
        ),
        help="Path to the Phase 1 manifest.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    return parser.parse_args()


def main() -> None:
    """Run the dataset splitting pipeline."""

    args = parse_args()

    df = load_manifest(args.manifest)
    usable_df, excluded_df = prepare_split_data(df)

    split_df = assign_splits(
        usable_df,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    verify_no_leakage(split_df)

    save_split_outputs(
        split_df,
        excluded_df,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print("\nDataset splitting completed.")
    print("Usable images:", len(split_df))
    print("Excluded images:", len(excluded_df))
    print("\nSplit counts:")
    print(split_df["split"].value_counts())


if __name__ == "__main__":
    main()

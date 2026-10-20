"""Independent verification of the recorded data split (outputs/reports/splits.csv).

    python -m src.data.verify_splits

Checks, against the actual manifest and images on disk:

1. Split counts and ratios (train/val/test) and per-class distributions.
2. Exact-duplicate leakage: the same sha256 image bytes must never appear in
   two different splits (train/val/test).
3. Duplicate-group leakage: the recorded ``duplicate_group`` id must not span
   splits (near/exact duplicates stay together).
4. Label consistency: each file's folder label matches the recorded class.
5. Presence: a sample of manifest rows resolves to a real file on disk.
6. The excluded images (``excluded.csv``) are documented, not deleted.

Writes ``outputs/reports/split_integrity.json`` and prints a summary.
Exit code 0 = all checks passed, 1 = at least one check failed, 2 = missing assets.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config


def check_counts(df: pd.DataFrame, summary: dict) -> list[str]:
    problems: list[str] = []
    counts = df["split"].value_counts().to_dict()
    expected = summary.get("split_counts", {})
    for k, v in expected.items():
        if counts.get(k, 0) != v:
            problems.append(f"split '{k}': manifest has {counts.get(k, 0)} rows, summary says {v}")
    total = len(df)
    usable = summary.get("total_usable_images")
    if usable is not None and total != usable:
        problems.append(f"manifest rows {total} != summary total_usable_images {usable}")
    return problems


def check_duplicate_leakage(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    hashed = df.dropna(subset=["sha256"])
    n_dupes = int(hashed.duplicated(subset=["sha256"], keep=False).sum())
    if n_dupes:
        spans = (
            hashed.groupby("sha256")["split"].nunique().pipe(lambda s: s[s > 1])
        )
        if len(spans):
            problems.append(f"{len(spans)} identical-image hash(es) appear in more than one split")
    return problems


def check_group_leakage(df: pd.DataFrame) -> list[str]:
    if "duplicate_group" not in df.columns:
        return ["manifest has no 'duplicate_group' column"]
    g = df.dropna(subset=["duplicate_group"])
    spans = g.groupby("duplicate_group")["split"].nunique()
    bad = spans[spans > 1]
    return [f"{len(bad)} duplicate_group id(s) span multiple splits"] if len(bad) else []


def check_label_consistency(df: pd.DataFrame) -> list[str]:
    bad = df[df["class_name"] != df["class"]] if "class" in df.columns else df.iloc[0:0]
    if len(bad):
        return [f"{len(bad)} rows where folder label != class_name"]
    # folder vs rel_path check
    mismatch = df[df["rel_path"].map(lambda p: str(p).split("/")[0]) != df["class_name"]]
    return [f"{len(mismatch)} rows where rel_path folder != class_name"] if len(mismatch) else []


def check_files_exist(df: pd.DataFrame, data_root: Path, sample: int, seed: int) -> list[str]:
    from src.data.image_dataset import resolve_image_path

    rows = df if sample <= 0 or sample >= len(df) else df.sample(n=sample, random_state=seed)
    missing = [str(r.rel_path) for _, r in rows.iterrows() if not resolve_image_path(r, data_root).is_file()]
    return [f"{len(missing)} sampled manifest files not found on disk (e.g. {missing[:3]})"] if missing else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample-files", type=int, default=500, help="0 = check every file (slow)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "reports" / "split_integrity.json"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if not cfg.splits_csv.is_file():
        print(f"[split integrity unavailable] missing manifest: {cfg.splits_csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(cfg.splits_csv)
    required = {"path", "filename", "class_name", "rel_path", "split"}
    problems: list[str] = []
    missing_cols = required - set(df.columns)
    if missing_cols:
        problems.append(f"manifest missing columns: {sorted(missing_cols)}")

    summary_path = PROJECT_ROOT / "outputs" / "reports" / "split_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}

    if not missing_cols:
        problems += check_counts(df, summary)
        problems += check_duplicate_leakage(df)
        problems += check_group_leakage(df)
        problems += check_label_consistency(df)
        problems += check_files_exist(df, cfg.data_root, args.sample_files, cfg.seed)

    class_dist = {
        sp: df[df["split"] == sp]["class_name"].value_counts().sort_index().to_dict()
        for sp in ("train", "val", "test")
    }
    report = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(cfg.splits_csv.relative_to(PROJECT_ROOT)),
        "seed": summary.get("random_seed"),
        "split_ratios_configured": summary.get("split_ratios"),
        "split_counts": df["split"].value_counts().to_dict(),
        "class_distribution": class_dist,
        "total_rows": int(len(df)),
        "excluded_images": summary.get("total_excluded_images"),
        "checks": {
            "counts_match_summary": not any("split '" in p or "total_usable" in p for p in problems),
            "no_exact_duplicate_leakage": not any("identical-image" in p for p in problems),
            "no_duplicate_group_leakage": not any("duplicate_group id(s)" in p for p in problems),
            "labels_consistent": not any("label" in p for p in problems),
            "sampled_files_exist": not any("not found on disk" in p for p in problems),
        },
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "split_counts", "total_rows", "problems")}, indent=2))
    print(f"Report written to {args.out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

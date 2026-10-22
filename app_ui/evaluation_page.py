
"""Section D: Evaluation Dashboard - held-out test metrics, confusion matrix, error analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app_ui import common
from src.config import Config


def _weak_classes(per_class: dict, class_names: list[str]) -> list[str]:
    """Return classes whose F1 score is more than 2 percentage points below the macro average."""
    f1s = {
        name: per_class[name]["f1-score"]
        for name in class_names
        if name in per_class and "f1-score" in per_class[name]
    }

    if not f1s:
        return []

    macro_average = sum(f1s.values()) / len(f1s)

    return sorted(
        [
            name
            for name, score in f1s.items()
            if score < macro_average - 0.02
        ],
        key=lambda name: f1s[name],
    )


def _misclassification_examples(
    cfg: Config, max_rows: int = 12
) -> pd.DataFrame | None:
    """Load the highest-confidence incorrect predictions."""
    preds = common.load_report_csv("test_predictions.csv")

    if (
        preds is None
        or "correct" not in preds.columns
        or "confidence" not in preds.columns
    ):
        return None

    # CSV files may represent Boolean values as strings.
    correct_values = preds["correct"]

    if correct_values.dtype == object:
        correct_values = (
            correct_values.astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
        )

    wrong = preds[correct_values == False]  # noqa: E712
    wrong = wrong.sort_values("confidence", ascending=False)

    return wrong.head(max_rows)


def _show_example_thumbnails(cfg: Config, examples: pd.DataFrame) -> None:
    """Display source images for selected prediction errors, when available."""
    if examples.empty:
        return

    if not {"path", "class_name", "predicted"}.issubset(examples.columns):
        st.info("Image thumbnails unavailable: prediction report lacks required columns.")
        return

    root = cfg.data_root
    cols = st.columns(min(6, len(examples)))

    for i, (_, row) in enumerate(examples.head(6).iterrows()):
        name = Path(str(row["path"])).name
        path = root / str(row["class_name"]) / name
        col = cols[i % len(cols)]
        caption = f"{row['class_name']} → {row['predicted']}"

        if path.is_file():
            col.image(str(path), caption=caption, width=140)
        else:
            col.caption(f"{name} (image not on disk)")


def render(cfg: Config) -> None:
    """Render the evaluation dashboard from saved evaluation reports."""
    st.title("Evaluation Dashboard")
    st.caption(
        "Measured on the held-out test split of the recorded 70/15/15 split "
        "(seed 123)."
    )

    metrics = common.load_report_json("mobilenet_finetuned_test_metrics.json")

    if metrics is None:
        st.error(
            "No evaluation report found. Run:\n\n"
            "```bash\n"
            "python -m src.evaluation.run_evaluation --calibrate\n"
            "```"
        )
        return

    st.markdown(
        f"**Checkpoint** `{metrics.get('checkpoint', 'best_mobilenet_finetuned.pt')}` · "
        f"sha256 `{str(metrics.get('checkpoint_sha256', ''))[:16]}…` · "
        f"**split** {metrics.get('split', 'test')} "
        f"({metrics['test_samples']:,} images) · "
        f"**evaluated** {metrics.get('evaluated_at', 'unknown')}"
    )

    # Overall test metrics.
    columns = st.columns(6)
    columns[0].metric("Accuracy", common.pct(metrics["test_accuracy"]))
    columns[1].metric("Macro precision", common.pct(metrics["macro_precision"]))
    columns[2].metric("Macro recall", common.pct(metrics["macro_recall"]))
    columns[3].metric("Macro F1", common.pct(metrics["macro_f1"]))
    columns[4].metric("Weighted F1", common.pct(metrics["weighted_f1"]))
    columns[5].metric(
        "ECE (15 bins)",
        f"{metrics.get('ece_15_bins', float('nan')):.4f}",
    )

    # Per-class metrics.
    st.subheader("Per-class metrics")

    per_class = metrics.get("per_class", {})
    class_names = metrics.get("class_names", list(per_class))

    if per_class:
        table = pd.DataFrame.from_dict(per_class, orient="index")

        required_columns = ["precision", "recall", "f1-score", "support"]
        missing_columns = [
            column for column in required_columns if column not in table.columns
        ]

        if missing_columns:
            st.warning(
                "Some per-class metrics are missing: "
                + ", ".join(missing_columns)
            )
        else:
            table = table[required_columns].copy()

            # Use Python format specifications rather than printf-style strings.
            # This prevents literal strings such as "%.4f" appearing in the UI.
            st.dataframe(
                table.style.format(
                    {
                        "precision": "{:.4f}",
                        "recall": "{:.4f}",
                        "f1-score": "{:.4f}",
                        "support": "{:,.0f}",
                    },
                    na_rep="—",
                ),
                width="stretch",
            )

        weak = _weak_classes(per_class, class_names)

        if weak:
            st.warning(
                "Classes weaker than the macro average by >2 F1 points: "
                + ", ".join(f"**{cfg.display_name(name)}**" for name in weak)
                + ". Inspect their confusions below."
            )
        else:
            st.success(
                "No class trails the macro-average F1 by more than 2 points."
            )
    else:
        st.warning("Per-class metrics are not available in the evaluation report.")

    # Confusion matrix.
    st.subheader("Confusion matrix")

    cm_png = common.REPORTS_DIR / "confusion_matrix.png"
    m1, m2 = st.columns([3, 2])

    if cm_png.is_file():
        m1.image(
            str(cm_png),
            caption="Row-normalised confusion matrix (PNG artifact)",
            width="stretch",
        )
    else:
        m1.info(
            "Run `python -m src.evaluation.run_evaluation` "
            "to generate confusion_matrix.png."
        )

    cm = metrics.get("confusion_matrix")

    if cm:
        with m2:
            st.caption("Raw counts")
            st.dataframe(
                pd.DataFrame(cm, index=class_names, columns=class_names),
                width="stretch",
            )

    st.caption(
        "Row-normalised view: each row sums to 1 over the true class; "
        "diagonal = recall of that class."
    )

    # Misclassification analysis.
    st.subheader("Misclassifications")

    conf = common.load_report_csv("top_confusions.csv")
    e1, e2 = st.columns([2, 3])

    if conf is not None and len(conf):
        e1.markdown("**Most frequent confusions**")
        e1.dataframe(conf, hide_index=True, width="stretch")
    else:
        e1.info("top_confusions.csv not available.")

    examples = _misclassification_examples(cfg)

    if examples is not None and len(examples):
        e2.markdown(
            f"**Example errors** (highest-confidence mistakes, "
            f"up to {len(examples)})"
        )
        e2.dataframe(examples, hide_index=True, width="stretch")
        _show_example_thumbnails(cfg, examples)
    else:
        e2.info("test_predictions.csv not available.")

    # Downloadable reports.
    st.subheader("Download reports")

    downloads = [
        (
            "mobilenet_finetuned_test_metrics.json",
            "metrics JSON",
            "application/json",
        ),
        (
            "mobilenet_finetuned_test_report.csv",
            "per-class CSV",
            "text/csv",
        ),
        (
            "mobilenet_finetuned_confusion_matrix.csv",
            "confusion matrix CSV",
            "text/csv",
        ),
        (
            "test_predictions.csv",
            "per-image predictions CSV",
            "text/csv",
        ),
        ("top_confusions.csv", "top confusions CSV", "text/csv"),
        ("confusion_matrix.png", "confusion matrix PNG", "image/png"),
    ]

    for name, label, mime in downloads:
        path = common.REPORTS_DIR / name

        if path.is_file():
            common.download_bytes(
                path.read_bytes(),
                name,
                f"Download {label}",
                mime,
            )

    # Research transparency.
    with st.expander("Measurement honesty"):
        st.markdown(
            "- Every number on this page comes from `outputs/reports/` files "
            "produced by `python -m src.evaluation.run_evaluation` on the "
            "recorded checkpoint and held-out split.\n"
            "- The test split is used only for final measurement, not for "
            "tuning or calibration. Temperature is fitted on the validation split.\n"
            "- The historical 97.07% claim was not reproduced by this "
            "checkpoint and is not shown as a result."
        )

"""Reproducible evaluation of the MobileNetV2 checkpoint on the held-out test split.

    python -m src.evaluation.run_evaluation            # test metrics + reports
    python -m src.evaluation.run_evaluation --calibrate  # also fit temperature on *val*

Outputs (outputs/reports/): mobilenet_finetuned_test_metrics.json,
mobilenet_finetuned_test_report.csv, mobilenet_finetuned_confusion_matrix.csv,
confusion_matrix.png, top_confusions.csv, test_predictions.csv, and
(with --calibrate) calibration.json.

The test split is only read here; no threshold or hyper-parameter is selected on it.
Missing assets (checkpoint, split manifest, images) produce a clear message and exit code 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.evaluation import calibration as cal
from src.evaluation import metrics as M
from src.inference.errors import CheckpointError
from src.utils.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--calibrate", action="store_true", help="fit a temperature on the validation split")
    ap.add_argument("--report-dir", default=str(PROJECT_ROOT / "outputs" / "reports"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    from src.evaluation.runner import collect_logits, make_dataset
    from src.inference.predictor import load_checkpoint

    try:
        loaded = load_checkpoint(cfg)
        dataset = make_dataset(args.split, cfg, loaded)
        # Fail early (before the full pass) if the images are not on disk.
        _ = dataset[0]
    except (CheckpointError, FileNotFoundError) as exc:
        print(f"[evaluation unavailable] {exc}", file=sys.stderr)
        return 2

    names = loaded.class_names
    print(f"Evaluating {len(dataset)} '{args.split}' images on CPU ...")
    logits, labels = collect_logits(loaded, dataset, args.batch_size)
    preds = logits.argmax(1)

    metrics = M.compute_metrics(labels, preds, names)
    probs = cal.softmax(logits)
    metrics.update(
        {
            "model": "MobileNetV2 fine-tuned",
            "split": args.split,
            "checkpoint_epoch": loaded.metadata.get("epoch"),
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "nll": cal.negative_log_likelihood(probs, labels),
            "ece_15_bins": cal.expected_calibration_error(probs, labels),
        }
    )

    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=[f"actual_{n}" for n in names], columns=[f"predicted_{n}" for n in names])
    M.per_class_table(metrics).to_csv(report_dir / "mobilenet_finetuned_test_report.csv")
    cm_df.to_csv(report_dir / "mobilenet_finetuned_confusion_matrix.csv")
    M.plot_confusion_matrix(metrics["confusion_matrix"], names, report_dir / "confusion_matrix.png")
    conf = M.top_confusions(metrics["confusion_matrix"], names)
    conf.to_csv(report_dir / "top_confusions.csv", index=False)

    pred_df = dataset.df[["path", "class_name"]].copy()
    pred_df["predicted"] = [names[i] for i in preds]
    pred_df["confidence"] = probs.max(1)
    pred_df["correct"] = preds == labels
    pred_df["path"] = pred_df["path"].map(lambda p: Path(str(p).replace("\\", "/")).name)  # no machine-specific paths
    pred_df.to_csv(report_dir / "test_predictions.csv", index=False)

    if args.calibrate:
        val_ds = make_dataset("val", cfg, loaded)
        v_logits, v_labels = collect_logits(loaded, val_ds, args.batch_size)
        t = cal.fit_temperature(v_logits, v_labels)
        p_t = cal.softmax(logits, t)
        calibration = {
            "temperature": t,
            "fit_split": "val",
            "checkpoint_sha256": loaded.checkpoint_sha256,
            f"{args.split}_nll_before": cal.negative_log_likelihood(probs, labels),
            f"{args.split}_nll_after": cal.negative_log_likelihood(p_t, labels),
            f"{args.split}_ece_before": cal.expected_calibration_error(probs, labels),
            f"{args.split}_ece_after": cal.expected_calibration_error(p_t, labels),
        }
        (report_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        metrics["calibration"] = calibration

    (report_dir / "mobilenet_finetuned_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nAccuracy        : {metrics['test_accuracy']:.4f}")
    print(f"Macro precision : {metrics['macro_precision']:.4f}")
    print(f"Macro recall    : {metrics['macro_recall']:.4f}")
    print(f"Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"NLL / ECE       : {metrics['nll']:.4f} / {metrics['ece_15_bins']:.4f}")
    print("\nTop confusions:\n" + conf.head(5).to_string(index=False))
    print(f"\nReports written to {report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Does watershed-based centre-cell cropping change classification accuracy?

    python -m src.experiments.segmentation_study --split val      # inspect / tune here
    python -m src.experiments.segmentation_study --split test     # report ONCE, parameters frozen

For each image: (a) baseline = whole image -> model; (b) cropped = square crop around
the object nearest the centre found by ``src.segmentation.watershed`` -> model
(falls back to the whole image if nothing is found). The model is NOT retrained.
Also records how many objects the watershed finds per class (a proxy for overlapping
or multi-cell images). Only measured numbers are written; no improvement is assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from src.config import PROJECT_ROOT, load_config
from src.utils.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--max-images", type=int, default=0, help="0 = all images")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "reports" / "segmentation_study.json"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    from src.data.image_dataset import resolve_image_path
    from src.evaluation.runner import make_dataset
    from src.inference.errors import CheckpointError
    from src.inference.predictor import Predictor, load_checkpoint
    from src.segmentation.watershed import central_object_crop, segment_cells

    try:
        loaded = load_checkpoint(cfg)
        ds = make_dataset(args.split, cfg, loaded)
        resolve_image_path(ds.df.iloc[0], cfg.data_root)
    except (CheckpointError, FileNotFoundError) as exc:
        print(f"[segmentation study unavailable] {exc}", file=sys.stderr)
        return 2

    predictor = Predictor(loaded, cfg)
    df = ds.df if args.max_images <= 0 else ds.df.sample(n=min(args.max_images, len(ds.df)), random_state=cfg.seed)
    rows = []
    for _, row in df.iterrows():
        image = Image.open(resolve_image_path(row, cfg.data_root)).convert("RGB")
        y = ds.class_to_idx[row["class_name"]]
        seg = segment_cells(image)
        crop = central_object_crop(image, seg, margin=args.margin)
        rows.append(
            {
                "class": row["class_name"],
                "n_objects": seg.n_objects,
                "crop_found": crop is not None,
                "base_correct": int(predictor.logits(image).argmax() == y),
                "crop_correct": int(predictor.logits(crop if crop is not None else image).argmax() == y),
            }
        )

    import pandas as pd

    r = pd.DataFrame(rows)
    multi = r[r.n_objects > 1]
    report = {
        "split": args.split,
        "n_images": int(len(r)),
        "baseline_accuracy": float(r.base_correct.mean()),
        "cropped_accuracy": float(r.crop_correct.mean()),
        "accuracy_difference_crop_minus_baseline": float(r.crop_correct.mean() - r.base_correct.mean()),
        "fraction_crop_found": float(r.crop_found.mean()),
        "fraction_multiple_objects": float((r.n_objects > 1).mean()),
        "multi_object_subset": {
            "n": int(len(multi)),
            "baseline_accuracy": float(multi.base_correct.mean()) if len(multi) else None,
            "cropped_accuracy": float(multi.crop_correct.mean()) if len(multi) else None,
        },
        "objects_per_image_by_class": r.groupby("class").n_objects.mean().round(3).to_dict(),
        "note": "Measured effect of inference-time cropping with the existing model; no retraining. Do not claim an improvement unless the difference is positive on the held-out test split with parameters chosen on val.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

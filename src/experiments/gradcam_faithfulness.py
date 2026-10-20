"""Quantitative sanity check for Grad-CAM: deletion test vs random deletion.

    python -m src.experiments.gradcam_faithfulness --split val --max-images 200

For each image the top-q% most-activated Grad-CAM pixels are replaced by the image
mean colour and the drop in the predicted-class probability is measured. If the
heat-maps are informative, deleting them should reduce the probability more than
deleting the same number of random pixels. Reports both drops; makes no claim
beyond what the numbers show.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT, load_config
from src.utils.seed import set_seed


def delete_pixels(image: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).copy()
    arr[mask] = arr.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return Image.fromarray(arr)


def top_fraction_mask(cam: np.ndarray, fraction: float) -> np.ndarray:
    k = max(1, int(round(fraction * cam.size)))
    thresh = np.partition(cam.ravel(), -k)[-k]
    return cam >= thresh


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "reports" / "gradcam_faithfulness.json"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    from src.data.image_dataset import resolve_image_path
    from src.evaluation.runner import make_dataset
    from src.explainability.gradcam import GradCAM
    from src.inference.errors import CheckpointError
    from src.inference.predictor import Predictor, load_checkpoint
    from src.inference.preprocess import resize_image, to_model_input

    try:
        loaded = load_checkpoint(cfg)
        ds = make_dataset(args.split, cfg, loaded)
        resolve_image_path(ds.df.iloc[0], cfg.data_root)
    except (CheckpointError, FileNotFoundError) as exc:
        print(f"[gradcam faithfulness unavailable] {exc}", file=sys.stderr)
        return 2

    predictor = Predictor(loaded, cfg)
    sub = ds.df.sample(n=min(args.max_images, len(ds.df)), random_state=cfg.seed)
    drops = {f: {"gradcam": [], "random": []} for f in args.fractions}

    def prob_of(image: Image.Image, cls: int) -> float:
        z = predictor.logits(image)
        z = z - z.max()
        return float(np.exp(z[cls]) / np.exp(z).sum())

    with GradCAM(loaded.model) as engine:
        for _, row in sub.iterrows():
            image = resize_image(Image.open(resolve_image_path(row, cfg.data_root)), loaded.image_size)
            x = to_model_input(image, loaded.image_size, loaded.mean, loaded.std, cfg.pixel_scale)
            cam, cls, _ = engine(x)
            base = prob_of(image, cls)
            for f in args.fractions:
                drops[f]["gradcam"].append(base - prob_of(delete_pixels(image, top_fraction_mask(cam, f)), cls))
                rand = np.zeros(cam.size, dtype=bool)
                rand[rng.choice(cam.size, size=max(1, int(round(f * cam.size))), replace=False)] = True
                drops[f]["random"].append(base - prob_of(delete_pixels(image, rand.reshape(cam.shape)), cls))

    report = {
        "split": args.split,
        "n_images": int(len(sub)),
        "mean_probability_drop": {
            f"{int(f * 100)}%": {k: float(np.mean(v)) for k, v in d.items()} for f, d in drops.items()
        },
        "note": "Larger drop for gradcam than random supports (does not prove) that heat-maps highlight evidence the model uses.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

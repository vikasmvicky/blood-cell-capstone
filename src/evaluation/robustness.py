"""Robustness checks: accuracy under image degradations and an optional OOD probe.

    python -m src.evaluation.robustness --split test --max-images 500
    python -m src.evaluation.robustness --ood-dir path/to/non_cell_images

Perturbations are applied to the *original* image before the normal preprocessing.
Results are measurements on this dataset only; they do not certify behaviour on
other microscopes, stains or labs. The OOD probe reports how well the max-softmax
score separates in-distribution test images from the images you supply (AUROC);
a value near 0.5 means the confidence score is not a usable OOD detector.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.config import PROJECT_ROOT, load_config
from src.utils.seed import set_seed

Perturbation = Callable[[Image.Image, np.random.Generator], Image.Image]


def gaussian_blur(radius: float) -> Perturbation:
    return lambda img, rng: img.filter(ImageFilter.GaussianBlur(radius))


def gaussian_noise(sigma: float) -> Perturbation:
    def f(img: Image.Image, rng: np.random.Generator) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        return Image.fromarray(np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8))
    return f


def jpeg_compress(quality: int) -> Perturbation:
    def f(img: Image.Image, rng: np.random.Generator) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return f


def brightness(factor: float) -> Perturbation:
    return lambda img, rng: ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)


def contrast(factor: float) -> Perturbation:
    return lambda img, rng: ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)


def downscale(factor: float) -> Perturbation:
    def f(img: Image.Image, rng: np.random.Generator) -> Image.Image:
        w, h = img.size
        small = img.convert("RGB").resize((max(8, int(w * factor)), max(8, int(h * factor))), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)
    return f


DEFAULT_PERTURBATIONS: dict[str, Perturbation] = {
    "none": lambda img, rng: img.convert("RGB"),
    "blur_r1": gaussian_blur(1.0),
    "blur_r3": gaussian_blur(3.0),
    "noise_s10": gaussian_noise(10),
    "noise_s25": gaussian_noise(25),
    "jpeg_q30": jpeg_compress(30),
    "jpeg_q10": jpeg_compress(10),
    "brightness_0.6": brightness(0.6),
    "brightness_1.5": brightness(1.5),
    "contrast_0.5": contrast(0.5),
    "downscale_0.25": downscale(0.25),
}


def auroc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """AUROC for separating in-distribution (should score higher) from OOD samples."""
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones(len(scores_in)), np.zeros(len(scores_out))])
    return float(roc_auc_score(y, np.concatenate([scores_in, scores_out])))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max-images", type=int, default=500, help="stratification-free random subset size (seeded)")
    ap.add_argument("--ood-dir", default=None, help="folder of non-cell images for the OOD probe")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "reports" / "robustness.json"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    from src.data.image_dataset import resolve_image_path
    from src.evaluation.runner import make_dataset
    from src.inference.errors import CheckpointError
    from src.inference.predictor import Predictor, load_checkpoint
    from src.inference.preprocess import load_image

    try:
        loaded = load_checkpoint(cfg)
        ds = make_dataset(args.split, cfg, loaded)
        resolve_image_path(ds.df.iloc[0], cfg.data_root)
    except (CheckpointError, FileNotFoundError) as exc:
        print(f"[robustness unavailable] {exc}", file=sys.stderr)
        return 2

    predictor = Predictor(loaded, cfg)
    idx = rng.permutation(len(ds.df))[: args.max_images]
    sub = ds.df.iloc[idx]
    results = {name: {"correct": 0, "n": 0} for name in DEFAULT_PERTURBATIONS}
    in_conf = []
    for _, row in sub.iterrows():
        image = Image.open(resolve_image_path(row, cfg.data_root)).convert("RGB")
        y = ds.class_to_idx[row["class_name"]]
        for name, fn in DEFAULT_PERTURBATIONS.items():
            logits = predictor.logits(fn(image, rng))
            results[name]["correct"] += int(logits.argmax() == y)
            results[name]["n"] += 1
            if name == "none":
                z = logits - logits.max()
                in_conf.append(float(np.exp(z).max() / np.exp(z).sum()))

    report = {
        "split": args.split,
        "n_images": int(len(sub)),
        "seed": cfg.seed,
        "accuracy_by_perturbation": {k: v["correct"] / v["n"] for k, v in results.items()},
    }

    if args.ood_dir:
        files = [p for p in sorted(Path(args.ood_dir).iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if not files:
            print(f"No PNG/JPEG images in {args.ood_dir}", file=sys.stderr)
        else:
            out_conf = []
            for p in files:
                z = predictor.logits(load_image(p, min_side=8))
                z = z - z.max()
                out_conf.append(float(np.exp(z).max() / np.exp(z).sum()))
            report["ood_probe"] = {
                "n_ood_images": len(files),
                "mean_confidence_in_distribution": float(np.mean(in_conf)),
                "mean_confidence_ood": float(np.mean(out_conf)),
                "auroc_max_softmax": auroc(np.array(in_conf), np.array(out_conf)),
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

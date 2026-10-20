"""Grad-CAM faithfulness: deletion and insertion curves vs random-pixel baselines.

    python -m src.experiments.gradcam_faithfulness --split val --max-images 60 --steps 20 --random-seeds 3

Protocol (defined here so the numbers are interpretable and reproducible):

* Images come from the requested split (default: validation). Nothing is tuned
  on the test split; running with ``--split test`` is a one-shot report.
* For each image the Grad-CAM heat-map for the *predicted* class is computed
  on the 224x224 model input (the class is held fixed for the whole curve).
* **Deletion**: start from the original image and progressively *replace* the
  most salient pixels (highest Grad-CAM value first) with the per-image mean
  colour. **Insertion**: start from a blank mean-colour image and progressively
  *insert* the original pixels, most salient first.
* **Random baseline**: the same procedure with a uniformly random pixel order,
  repeated over ``--random-seeds`` independent orderings and averaged.
* At each of ``--steps`` fractions the target-class softmax probability is
  measured; curves are summarised by their area under the curve (AUC, trapezoid
  over the fraction grid, normalised to [0, 1]).
* Interpretation: a faithful saliency map should yield a *lower* deletion AUC
  and a *higher* insertion AUC than the random baseline. This supports - but
  never proves - that the heat-map highlights evidence the model uses.

Outputs: ``gradcam_faithfulness.json`` (machine-readable, mean +/- std over
images) and ``gradcam_faithfulness_curves.png`` (mean curves +/- std bands).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT, load_config
from src.utils.seed import set_seed


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in tests/test_faithfulness.py)
# --------------------------------------------------------------------------- #

def top_fraction_mask(cam: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask of the ``fraction`` highest-valued entries (ties kept)."""
    k = max(1, int(round(fraction * cam.size)))
    thresh = np.partition(cam.ravel(), -k)[-k]
    return cam >= thresh


def mean_colour_baseline(image_arr: np.ndarray) -> np.ndarray:
    """Constant image filled with the per-image mean colour."""
    flat = image_arr.reshape(-1, image_arr.shape[-1])
    return np.broadcast_to(flat.mean(axis=0), image_arr.shape).copy()


def perturb(image_arr: np.ndarray, order: np.ndarray, k: int, mode: str) -> np.ndarray:
    """Return the image with the first ``k`` pixels of flat ``order`` applied.

    mode='deletion': original image with those pixels replaced by mean colour.
    mode='insertion': mean-colour image with those original pixels inserted.
    """
    if mode not in {"deletion", "insertion"}:
        raise ValueError(f"mode must be 'deletion' or 'insertion', got {mode!r}")
    h, w, c = image_arr.shape
    out = image_arr.copy() if mode == "deletion" else mean_colour_baseline(image_arr)
    if k <= 0:
        return image_arr.copy() if mode == "deletion" else mean_colour_baseline(image_arr)
    if k >= order.size:
        return mean_colour_baseline(image_arr) if mode == "deletion" else image_arr.copy()
    flat = out.reshape(-1, c)
    base_flat = image_arr.reshape(-1, c)
    idx = order[:k]
    if mode == "deletion":
        flat[idx] = mean_colour_baseline(image_arr).reshape(-1, c)[idx]
    else:
        flat[idx] = base_flat[idx]
    return flat.reshape(h, w, c)


def curve_auc(values: np.ndarray, fractions: np.ndarray) -> float:
    """Trapezoidal AUC of a curve over the fraction grid (x in [0, 1])."""
    values = np.asarray(values, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    if values.size != fractions.size or values.size < 2:
        raise ValueError("values and fractions must have equal length >= 2")
    return float(np.trapezoid(values, fractions))


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #

def _to_norm_tensor(arrays: np.ndarray, mean, std) -> "np.ndarray":
    """HxWx3 float arrays in [0,1] -> (N,3,H,W) normalised (numpy, then torch in caller)."""
    x = arrays.transpose(0, 3, 1, 2)
    m = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    return ((x - m) / s).astype(np.float32)


def probabilities_for_steps(model, step_images: np.ndarray, cls: int, mean, std, batch_size: int = 32) -> np.ndarray:
    """Target-class probability for each perturbed image (batched forwards)."""
    import torch

    probs: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(step_images), batch_size):
            x = torch.from_numpy(_to_norm_tensor(step_images[start : start + batch_size], mean, std))
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, cls].cpu().numpy()
            probs.extend(float(v) for v in p)
    return np.asarray(probs)


def run_image_curves(
    image_arr: np.ndarray,
    cam: np.ndarray,
    cls: int,
    model,
    mean,
    std,
    fractions: np.ndarray,
    rngs: list[np.random.Generator],
) -> dict:
    """Curves for one image. ``rngs`` = one generator per random baseline seed."""
    h, w, c = image_arr.shape
    n_pix = h * w
    gradcam_order = np.argsort(-cam.ravel(), kind="stable")
    ks = np.clip((fractions * n_pix).round().astype(int), 0, n_pix)

    def stack_for(order: np.ndarray, mode: str) -> np.ndarray:
        return np.stack([perturb(image_arr, order, int(k), mode) for k in ks])

    out: dict = {
        "gradcam": {},
        "random": {i: {} for i in range(len(rngs))},
    }
    base = mean_colour_baseline(image_arr)
    for mode in ("deletion", "insertion"):
        start = image_arr if mode == "deletion" else base
        # k=0 endpoint for insertion is the blank image; for deletion it is the original
        imgs = [start] + [perturb(image_arr, gradcam_order, int(k), mode) for k in ks[1:]]
        out["gradcam"][mode] = probabilities_for_steps(model, np.stack(imgs), cls, mean, std)
        for i, rng in enumerate(rngs):
            order = rng.permutation(n_pix)
            imgs = [start] + [perturb(image_arr, order, int(k), mode) for k in ks[1:]]
            out["random"][i][mode] = probabilities_for_steps(model, np.stack(imgs), cls, mean, std)
    return out


def aggregate_results(all_curves: list[dict], fractions: np.ndarray) -> dict:
    """Mean/std curves and AUCs across images; random seeds averaged per image first."""
    gc_del = np.array([c["gradcam"]["deletion"] for c in all_curves])
    gc_ins = np.array([c["gradcam"]["insertion"] for c in all_curves])
    rnd_del = np.array([[c["random"][i]["deletion"] for i in sorted(c["random"])] for c in all_curves])
    rnd_ins = np.array([[c["random"][i]["insertion"] for i in sorted(c["random"])] for c in all_curves])
    rnd_del_mean, rnd_ins_mean = rnd_del.mean(axis=1), rnd_ins.mean(axis=1)  # (n_images, steps+1)

    def stats(arr: np.ndarray) -> dict:
        return {"mean": arr.mean(axis=0).round(6).tolist(), "std": arr.std(axis=0).round(6).tolist()}

    aucs = {
        "gradcam_deletion": np.array([curve_auc(v, fractions) for v in gc_del]),
        "gradcam_insertion": np.array([curve_auc(v, fractions) for v in gc_ins]),
        "random_deletion": np.array([curve_auc(v, fractions) for v in rnd_del_mean]),
        "random_insertion": np.array([curve_auc(v, fractions) for v in rnd_ins_mean]),
    }
    return {
        "fractions": fractions.tolist(),
        "curves": {
            "gradcam_deletion": stats(gc_del),
            "gradcam_insertion": stats(gc_ins),
            "random_deletion": stats(rnd_del_mean),
            "random_insertion": stats(rnd_ins_mean),
        },
        "auc": {
            k: {
                "mean": float(v.mean()),
                "std": float(v.std()),
                "values": v.round(6).tolist(),
            }
            for k, v in aucs.items()
        },
    }


def plot_curves(agg: dict, out_path: Path) -> None:
    """Two panels (deletion, insertion): Grad-CAM vs random mean +/- std."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(agg["fractions"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [
        ("Deletion (lower = more faithful)", "gradcam_deletion", "random_deletion", None),
        ("Insertion (higher = more faithful)", "gradcam_insertion", "random_insertion", None),
    ]
    for ax, (title, gk, rk, _) in zip(axes, panels):
        for key, label, color in ((gk, "Grad-CAM", "#d62728"), (rk, "Random", "#7f7f7f")):
            mean = np.asarray(agg["curves"][key]["mean"])
            std = np.asarray(agg["curves"][key]["std"])
            ax.plot(x, mean, label=label, color=color)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        auc_g = agg["auc"][gk]["mean"]
        auc_r = agg["auc"][rk]["mean"]
        ax.set_title(f"{title}\nAUC Grad-CAM {auc_g:.3f} vs random {auc_r:.3f}")
        ax.set_xlabel("Fraction of pixels perturbed")
        ax.set_ylabel("Target-class probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--max-images", type=int, default=60)
    ap.add_argument("--steps", type=int, default=20, help="perturbation steps between 0 and 100%")
    ap.add_argument("--random-seeds", type=int, default=3, help="random-order baseline repeats per image")
    ap.add_argument("--seed", type=int, default=None, help="sampling seed (default: config seed)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "reports" / "gradcam_faithfulness.json"))
    ap.add_argument("--plot", default=str(PROJECT_ROOT / "outputs" / "reports" / "gradcam_faithfulness_curves.png"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed)
    rng = np.random.default_rng(seed)

    from src.data.image_dataset import resolve_image_path
    from src.evaluation.runner import make_dataset
    from src.explainability.gradcam import GradCAM
    from src.inference.errors import CheckpointError
    from src.inference.predictor import load_checkpoint
    from src.inference.preprocess import image_to_array

    try:
        loaded = load_checkpoint(cfg)
        ds = make_dataset(args.split, cfg, loaded)
        resolve_image_path(ds.df.iloc[0], cfg.data_root)
    except (CheckpointError, FileNotFoundError) as exc:
        print(f"[gradcam faithfulness unavailable] {exc}", file=sys.stderr)
        return 2

    model = loaded.model
    fractions = np.linspace(0.0, 1.0, args.steps + 1)
    rngs = [np.random.default_rng(seed + 1000 + i) for i in range(args.random_seeds)]
    sub = ds.df.sample(n=min(args.max_images, len(ds.df)), random_state=seed)

    all_curves: list[dict] = []
    target_classes: list[int] = []
    for n, (_, row) in enumerate(sub.iterrows(), start=1):
        image = Image.open(resolve_image_path(row, cfg.data_root)).convert("RGB")
        arr = image_to_array(image, loaded.image_size, cfg.pixel_scale)  # [0,1] HWC
        import torch

        x = torch.from_numpy(_to_norm_tensor(arr[None], loaded.mean, loaded.std))
        with GradCAM(model) as engine:
            cam, cls, _ = engine(x)
        target_classes.append(int(cls))
        all_curves.append(run_image_curves(arr, cam, cls, model, loaded.mean, loaded.std, fractions, rngs))
        if n % 10 == 0 or n == len(sub):
            print(f"  {n}/{len(sub)} images done", flush=True)

    agg = aggregate_results(all_curves, fractions)
    del_auc = agg["auc"]["gradcam_deletion"]["mean"]
    ins_auc = agg["auc"]["gradcam_insertion"]["mean"]
    rnd_del_auc = agg["auc"]["random_deletion"]["mean"]
    rnd_ins_auc = agg["auc"]["random_insertion"]["mean"]

    report = {
        "experiment": "gradcam_deletion_insertion_curves",
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "n_images": int(len(sub)),
        "steps": args.steps,
        "random_baseline_seeds": args.random_seeds,
        "sampling_seed": seed,
        "target_class": "predicted (argmax) class of the unperturbed image",
        "baseline_image": "per-image mean colour (RGB mean of the 224x224 input)",
        "perturbation": "pixels replaced/inserted individually in Grad-CAM order at 224x224 resolution; softmax probability of the fixed target class measured after each step",
        "checkpoint": str(loaded.checkpoint_path.name),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        **agg,
        "summary": {
            "gradcam_deletion_auc": round(del_auc, 4),
            "random_deletion_auc": round(rnd_del_auc, 4),
            "gradcam_insertion_auc": round(ins_auc, 4),
            "random_insertion_auc": round(rnd_ins_auc, 4),
            "deletion_more_faithful_than_random": bool(del_auc < rnd_del_auc),
            "insertion_more_faithful_than_random": bool(ins_auc > rnd_ins_auc),
        },
        "note": (
            "Lower deletion AUC and higher insertion AUC for Grad-CAM than for random pixel orderings "
            "support (do not prove) that the heat-maps highlight evidence the model actually uses. "
            "Curves are means over the sampled images; per-image values are in 'auc'."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.plot:
        plot_curves(report, Path(args.plot))
        print(f"Plot written to {args.plot}")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

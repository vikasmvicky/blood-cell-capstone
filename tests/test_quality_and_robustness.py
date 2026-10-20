import numpy as np
from PIL import Image

from src.evaluation.robustness import DEFAULT_PERTURBATIONS, auroc
from src.inference.quality import check_quality, compute_quality_stats


def codes(img):
    return {w.code for w in check_quality(img)}


def test_quality_flags_dark_bright_flat_and_nonsquare():
    assert "dark" in codes(Image.new("RGB", (224, 224), (5, 5, 5)))
    assert "bright" in codes(Image.new("RGB", (224, 224), (250, 250, 250)))
    flat = codes(Image.new("RGB", (224, 224), (200, 120, 180)))
    assert {"low_contrast", "blurry"} <= flat
    assert "aspect_ratio" in codes(Image.new("RGB", (400, 200), (200, 120, 180)))
    assert "greyscale" in codes(Image.new("RGB", (224, 224), (128, 128, 128)))


def test_quality_stats_keys():
    s = compute_quality_stats(Image.new("RGB", (100, 50)))
    assert s["aspect_ratio"] == 2.0 and {"mean_brightness", "laplacian_var", "mean_saturation"} <= set(s)


def test_all_perturbations_return_same_size_rgb():
    rng = np.random.default_rng(0)
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (96, 96, 3), dtype=np.uint8))
    for name, fn in DEFAULT_PERTURBATIONS.items():
        out = fn(img, rng)
        assert out.mode == "RGB" and out.size == img.size, name


def test_noise_and_blur_change_pixels():
    rng = np.random.default_rng(0)
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    for name in ("blur_r3", "noise_s25", "brightness_0.6"):
        assert not np.array_equal(np.asarray(DEFAULT_PERTURBATIONS[name](img, rng)), np.asarray(img))


def test_auroc_perfect_and_chance():
    assert auroc(np.array([0.9, 0.95]), np.array([0.2, 0.3])) == 1.0
    assert auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5

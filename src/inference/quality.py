"""Heuristic input-quality checks shown as warnings next to a prediction.

IMPORTANT: these are simple, *unvalidated* heuristics (brightness, contrast,
blur, colour saturation, aspect ratio). They flag images that look unlike a
typical stained PBC smear crop; they are **not** an out-of-distribution detector
and do not guarantee anything about prediction correctness. Thresholds are
module constants so they can be calibrated against the real dataset with
``compute_quality_stats``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

MIN_MEAN_BRIGHTNESS = 40.0     # 0-255 grey level; below -> "too dark"
MAX_MEAN_BRIGHTNESS = 235.0    # above -> "too bright / washed out"
MIN_CONTRAST_STD = 12.0        # grey-level std; below -> "low contrast"
MIN_LAPLACIAN_VAR = 20.0       # on a 224px grey image; below -> "possibly blurry"
MIN_MEAN_SATURATION = 0.06     # HSV saturation 0-1; below -> "almost greyscale"
MAX_ASPECT_RATIO = 1.25        # long/short side; above -> distorted by resize


@dataclass(frozen=True)
class QualityWarning:
    code: str
    message: str


def compute_quality_stats(image: Image.Image) -> dict[str, float]:
    """Numeric statistics behind the warnings (useful for calibrating thresholds)."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    small = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    grey = np.asarray(small.convert("L"), dtype=np.float32)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(max(width, height) / max(1, min(width, height))),
        "mean_brightness": float(grey.mean()),
        "contrast_std": float(grey.std()),
        "laplacian_var": float(ndimage.laplace(grey).var()),
        "mean_saturation": float(hsv[..., 1].mean() / 255.0),
    }


def check_quality(image: Image.Image) -> list[QualityWarning]:
    """Return human-readable warnings (empty list = nothing unusual detected)."""
    s = compute_quality_stats(image)
    out: list[QualityWarning] = []
    if s["mean_brightness"] < MIN_MEAN_BRIGHTNESS:
        out.append(QualityWarning("dark", "The image is very dark."))
    if s["mean_brightness"] > MAX_MEAN_BRIGHTNESS:
        out.append(QualityWarning("bright", "The image is very bright or washed out."))
    if s["contrast_std"] < MIN_CONTRAST_STD:
        out.append(QualityWarning("low_contrast", "The image has very low contrast."))
    if s["laplacian_var"] < MIN_LAPLACIAN_VAR:
        out.append(QualityWarning("blurry", "The image may be blurry or out of focus."))
    if s["mean_saturation"] < MIN_MEAN_SATURATION:
        out.append(QualityWarning("greyscale", "The image has almost no colour; the model was trained on stained colour smears."))
    if s["aspect_ratio"] > MAX_ASPECT_RATIO:
        out.append(QualityWarning("aspect_ratio", "The image is not roughly square; resizing to 224x224 will distort it. Crop around a single cell for best results."))
    return out

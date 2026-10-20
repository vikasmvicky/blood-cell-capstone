"""Classical (training-free) cell segmentation with distance-transform watershed.

Purpose in this project: a *diagnostic / experimental* module. It is not part of
the classification pipeline and nothing here is claimed to improve accuracy.
``src/experiments/segmentation_study.py`` measures whether segmentation-based
cropping changes accuracy of the existing model on real data.

Method: grey/green-channel Otsu threshold -> morphological clean-up -> Euclidean
distance transform -> local maxima as markers -> watershed to split touching
objects. Parameters were NOT tuned on the PBC data in this repository; the unit
tests only verify the mechanics on synthetic discs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max


@dataclass
class SegmentationResult:
    mask: np.ndarray            # bool HxW, foreground
    labels: np.ndarray          # int HxW, 0 = background, 1..N = objects
    n_objects: int
    areas: list[int]
    threshold: float

    def overlay(self, image: Image.Image) -> Image.Image:
        """Return the image with object boundaries drawn in green."""
        rgb = np.asarray(image.convert("RGB").resize(self.labels.shape[::-1], Image.Resampling.BILINEAR)).copy()
        edges = segmentation.find_boundaries(self.labels, mode="outer")
        rgb[edges] = (0, 255, 0)
        return Image.fromarray(rgb)


def _drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    lab = measure.label(mask)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab]


def segment_cells(
    image: Image.Image,
    *,
    min_area_fraction: float = 0.002,
    min_peak_distance: int = 7,
    smooth_sigma: float = 1.5,
    foreground: str = "dark",
) -> SegmentationResult:
    """Segment and split objects. ``foreground='dark'`` treats stained cells as darker than background."""
    if foreground not in {"dark", "bright"}:
        raise ValueError("foreground must be 'dark' or 'bright'")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    channel = filters.gaussian(rgb[..., 1] / 255.0, sigma=smooth_sigma)  # green channel: stain absorbs it
    if np.ptp(channel) < 1e-6:  # flat image -> no objects
        empty = np.zeros(channel.shape, dtype=np.int32)
        return SegmentationResult(empty.astype(bool), empty, 0, [], float(channel.mean()))

    thr = float(filters.threshold_otsu(channel))
    mask = channel < thr if foreground == "dark" else channel > thr

    min_area = max(16, int(min_area_fraction * mask.size))
    mask = _drop_small_components(mask, min_area)   # version-independent (skimage renamed its arg)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, structure=morphology.disk(2))
    if not mask.any():
        empty = np.zeros(mask.shape, dtype=np.int32)
        return SegmentationResult(mask, empty, 0, [], thr)

    distance = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(distance, min_distance=min_peak_distance, labels=measure.label(mask), exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i
    labels = segmentation.watershed(-distance, markers, mask=mask)

    areas = [int(r.area) for r in measure.regionprops(labels) if r.area >= min_area]
    keep = {r.label for r in measure.regionprops(labels) if r.area >= min_area}
    labels = np.where(np.isin(labels, list(keep)), labels, 0).astype(np.int32)
    labels, _, _ = segmentation.relabel_sequential(labels)
    return SegmentationResult(mask, labels, int(labels.max()), areas, thr)


def central_object_crop(image: Image.Image, result: SegmentationResult, margin: float = 0.15) -> Image.Image | None:
    """Square crop around the object nearest the image centre (None if nothing found)."""
    if result.n_objects == 0:
        return None
    h, w = result.labels.shape
    centre = np.array([h / 2, w / 2])
    best, best_dist = None, float("inf")
    for region in measure.regionprops(result.labels):
        d = float(np.linalg.norm(np.array(region.centroid) - centre))
        if d < best_dist:
            best, best_dist = region, d
    r0, c0, r1, c1 = best.bbox
    side = int(max(r1 - r0, c1 - c0) * (1 + 2 * margin))
    cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
    top, left = int(round(cy - side / 2)), int(round(cx - side / 2))
    box = (max(0, left), max(0, top), min(w, left + side), min(h, top + side))
    img = image.convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.Resampling.BILINEAR)
    return img.crop(box)

import numpy as np
from PIL import Image

from src.segmentation.watershed import central_object_crop, segment_cells


def discs(centres, radius=28, size=224):
    yy, xx = np.mgrid[:size, :size]
    arr = np.full((size, size, 3), 235, dtype=np.uint8)
    for cy, cx in centres:
        arr[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = (120, 40, 140)
    return Image.fromarray(arr)


def test_single_disc_is_one_object():
    res = segment_cells(discs([(112, 112)]))
    assert res.n_objects == 1 and res.mask.any()


def test_overlapping_discs_are_split():
    res = segment_cells(discs([(112, 90), (112, 135)]))   # centres 45px apart, radius 28 -> overlap
    assert res.n_objects == 2


def test_separate_discs_counted_and_empty_image():
    assert segment_cells(discs([(50, 50), (170, 170)])).n_objects == 2
    assert segment_cells(Image.new("RGB", (224, 224), (230, 230, 230))).n_objects == 0


def test_central_crop_picks_nearest_object_and_overlay():
    img = discs([(112, 112), (30, 30)])
    res = segment_cells(img)
    crop = central_object_crop(img, res)
    assert crop is not None and 40 < crop.size[0] < 120
    assert res.overlay(img).size == img.size
    assert central_object_crop(img, segment_cells(Image.new("RGB", (224, 224), (230, 230, 230)))) is None

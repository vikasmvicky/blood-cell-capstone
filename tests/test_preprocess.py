import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.image_dataset import BloodCellDataset
from src.inference.errors import InvalidImageError
from src.inference.preprocess import load_image, to_model_input
from tests.conftest import png_bytes


def test_tensor_shape_and_dtype():
    t = to_model_input(Image.new("RGB", (300, 200), (10, 20, 30)))
    assert tuple(t.shape) == (1, 3, 224, 224) and t.dtype == torch.float32


def test_inference_preprocessing_matches_training_dataset(synthetic_assets):
    """Same image through the dataset path (+ trainer normalisation) and the inference path."""
    ds = BloodCellDataset("test", csv_path=synthetic_assets.splits_csv, class_map_path=synthetic_assets.class_map_path, data_root=synthetic_assets.data_root)
    from src.data.image_dataset import resolve_image_path

    img_tensor, _ = ds[0]
    mean = torch.tensor(synthetic_assets.mean).view(3, 1, 1)
    std = torch.tensor(synthetic_assets.std).view(3, 1, 1)
    expected = ((img_tensor - mean) / std).unsqueeze(0)
    pil = Image.open(resolve_image_path(ds.df.iloc[0], synthetic_assets.data_root)).convert("RGB")
    got = to_model_input(pil, synthetic_assets.image_size, synthetic_assets.mean, synthetic_assets.std)
    assert torch.allclose(got, expected, atol=1e-6)


def test_load_valid_png_and_jpeg():
    assert load_image(png_bytes()).mode == "RGB"
    assert load_image(png_bytes(fmt="JPEG")).size == (120, 120)


def test_transparent_png_composited_on_white():
    buf = io.BytesIO()
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(buf, format="PNG")
    assert np.asarray(load_image(buf.getvalue())).min() == 255


@pytest.mark.parametrize(
    "data,match",
    [
        (b"", "empty"),
        (b"this is not an image at all", "could not be read"),
        (png_bytes()[:60], "could not be read|truncated"),          # corrupted / truncated
    ],
)
def test_invalid_files_rejected(data, match):
    with pytest.raises(InvalidImageError, match=match):
        load_image(data)


def test_unsupported_format_too_small_too_large_oversize():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, format="GIF")
    with pytest.raises(InvalidImageError, match="Unsupported"):
        load_image(buf.getvalue())
    with pytest.raises(InvalidImageError, match="too small"):
        load_image(png_bytes(size=(20, 20)))
    with pytest.raises(InvalidImageError, match="too large"):
        load_image(png_bytes(size=(300, 100)), max_side=200)
    with pytest.raises(InvalidImageError, match="limit"):
        load_image(png_bytes(), max_bytes=100)

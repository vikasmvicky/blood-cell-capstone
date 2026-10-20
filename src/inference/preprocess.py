"""Image loading and model-input preprocessing for inference.

The pipeline mirrors training exactly (see ``src/data/image_dataset.py`` and the
trainers):  RGB -> PIL bilinear resize to 224x224 -> /255 -> ImageNet mean/std.
``tests/test_preprocess.py`` asserts that this path and the dataset path give
identical tensors.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.inference.errors import InvalidImageError

ALLOWED_FORMATS = {"PNG", "JPEG"}


def load_image(
    source: bytes | str | Path | BinaryIO,
    *,
    max_bytes: int | None = 10 * 1024 * 1024,
    min_side: int = 64,
    max_side: int = 8000,
) -> Image.Image:
    """Decode and validate an image, returning a fully-loaded RGB ``PIL.Image``.

    Raises :class:`InvalidImageError` with a readable message for empty,
    oversized, corrupt, unsupported or implausibly small/large files.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise InvalidImageError(f"File not found: {path.name}")
        data = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        data = source.read()

    if not data:
        raise InvalidImageError("The file is empty.")
    if max_bytes is not None and len(data) > max_bytes:
        raise InvalidImageError(
            f"The file is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB."
        )

    # Pass 1: verify structure (verify() invalidates the handle, so reopen after).
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(
            "The file could not be read as an image. It may be corrupt or not an image."
        ) from exc
    except Image.DecompressionBombError as exc:  # pragma: no cover - defensive
        raise InvalidImageError("The image has too many pixels to process safely.") from exc

    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError(
            f"Unsupported image format '{fmt}'. Please upload a PNG or JPEG image."
        )

    # Pass 2: full decode (catches truncated files that verify() can miss).
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = img.size
            if min(width, height) < min_side:
                raise InvalidImageError(
                    f"The image is too small ({width}x{height}px); at least {min_side}px per side is required."
                )
            if max(width, height) > max_side:
                raise InvalidImageError(
                    f"The image is too large ({width}x{height}px); the maximum is {max_side}px per side."
                )
            return _to_rgb(img)
    except InvalidImageError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The image is truncated or corrupt and could not be decoded.") from exc


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB; composite transparent PNGs over white instead of black."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def resize_image(image: Image.Image, image_size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Same resize as the training dataset (PIL bilinear, no aspect preservation)."""
    return image.convert("RGB").resize(tuple(image_size), Image.Resampling.BILINEAR)


def image_to_array(image: Image.Image, image_size: tuple[int, int] = (224, 224), pixel_scale: float = 255.0) -> np.ndarray:
    """Return a float32 HxWx3 array in [0, 1] (un-normalised)."""
    return np.asarray(resize_image(image, image_size), dtype=np.float32) / pixel_scale


def to_model_input(
    image: Image.Image,
    image_size: tuple[int, int] = (224, 224),
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    pixel_scale: float = 255.0,
):
    """Return a normalised ``(1, 3, H, W)`` float32 tensor ready for the model."""
    import torch

    arr = image_to_array(image, image_size, pixel_scale).transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t

"""UI-independent glue used by the Streamlit app (and testable without Streamlit)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.config import Config
from src.inference.predictor import Predictor, Prediction, load_checkpoint, load_temperature
from src.inference.preprocess import load_image


def build_predictor(cfg: Config) -> Predictor:
    """Load the checkpoint (raises CheckpointError subclasses) and optional temperature."""
    loaded = load_checkpoint(cfg)
    temperature = load_temperature(cfg.model_path.parent.parent / "reports", loaded.checkpoint_sha256)
    return Predictor(loaded, cfg, temperature=temperature)


def validate_upload(data: bytes, filename: str, cfg: Config) -> Image.Image:
    """Check extension, size and decodability; raise InvalidImageError otherwise."""
    from src.inference.errors import InvalidImageError

    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in cfg.allowed_extensions:
        raise InvalidImageError(f"Unsupported file type '.{ext}'. Allowed: {', '.join(cfg.allowed_extensions)}.")
    return load_image(
        data,
        max_bytes=int(cfg.max_upload_mb * 1024 * 1024),
        min_side=cfg.min_image_side_px,
        max_side=cfg.max_image_side_px,
    )


def predict_upload(data: bytes, filename: str, predictor: Predictor, cfg: Config, with_gradcam: bool = True) -> tuple[Image.Image, Prediction]:
    image = validate_upload(data, filename, cfg)
    return image, predictor.predict(image, with_gradcam=with_gradcam)

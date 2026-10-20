"""Checkpoint loading and single-image prediction (CPU-friendly)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.config import Config, class_names_from_mapping, load_class_names
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.inference.errors import CheckpointIncompatibleError, CheckpointNotFoundError
from src.inference.preprocess import to_model_input
from src.inference.quality import QualityWarning, check_quality
from src.models.mobilenet import build_mobilenet_v2

CALIBRATION_FILENAME = "calibration.json"


@dataclass
class LoadedModel:
    model: torch.nn.Module
    class_names: list[str]
    image_size: tuple[int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    checkpoint_path: Path
    checkpoint_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    class_name: str
    display_name: str
    class_index: int
    confidence: float                       # top-1 softmax score (optionally temperature-scaled)
    ranked: list[tuple[str, float]]         # (class_name, probability), descending
    probability_kind: str                   # "raw softmax" or "temperature-scaled softmax"
    margin: float                           # top1 - top2 probability
    normalized_entropy: float               # 0 (certain) .. 1 (uniform)
    low_confidence: bool
    quality_warnings: list[QualityWarning] = field(default_factory=list)
    cam: np.ndarray | None = None
    overlay: Image.Image | None = None
    gradcam_error: str | None = None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise_image_size(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    value = tuple(int(v) for v in value)
    return (value[0], value[0]) if len(value) == 1 else (value[0], value[1])


def load_checkpoint(cfg: Config, checkpoint_path: Path | None = None, device: str | None = None) -> LoadedModel:
    """Load the trained MobileNetV2 checkpoint, validating it strictly.

    Never falls back to random weights. Raises ``CheckpointNotFoundError`` when the
    file is absent and ``CheckpointIncompatibleError`` when it does not match.
    """
    path = Path(checkpoint_path or cfg.model_path)
    if not path.is_file():
        raise CheckpointNotFoundError(
            f"Trained checkpoint not found at '{path}'. Copy 'best_mobilenet_finetuned.pt' "
            f"into that location (or set BLOOD_CELL_MODEL_PATH). See ASSETS.md."
        )

    try:
        # weights_only=True: the file may only contain tensors / plain containers.
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - surface any unpickling/IO problem uniformly
        raise CheckpointIncompatibleError(f"Could not read checkpoint '{path.name}': {exc}") from exc

    notes: list[str] = []
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict, meta = ckpt["model_state_dict"], {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    elif isinstance(ckpt, dict) and any(k.startswith("features.") for k in ckpt):
        state_dict, meta = ckpt, {}
        notes.append("Checkpoint is a bare state_dict without metadata; class order taken from class_to_idx.json.")
    else:
        raise CheckpointIncompatibleError("Checkpoint format not recognised (expected 'model_state_dict').")

    # Class mapping: checkpoint metadata wins; it must agree with class_to_idx.json when that exists.
    if "class_to_idx" in meta:
        class_names = class_names_from_mapping(meta["class_to_idx"])
        if cfg.class_map_path.is_file():
            expected = load_class_names(cfg.class_map_path)
            if expected != class_names:
                raise CheckpointIncompatibleError(
                    f"Class order in checkpoint {class_names} differs from {cfg.class_map_path.name} {expected}."
                )
    elif cfg.class_map_path.is_file():
        class_names = load_class_names(cfg.class_map_path)
    else:
        raise CheckpointIncompatibleError("No class mapping available in checkpoint or class_to_idx.json.")

    image_size = _normalise_image_size(meta.get("image_size"), cfg.image_size)
    norm = meta.get("normalization") or {}
    mean = tuple(norm.get("mean", cfg.mean))
    std = tuple(norm.get("std", cfg.std))
    if image_size != cfg.image_size:
        notes.append(f"Using checkpoint image size {image_size} (config says {cfg.image_size}).")
    if mean != tuple(cfg.mean) or std != tuple(cfg.std):
        notes.append("Using normalisation statistics stored in the checkpoint (differ from config).")

    model = build_mobilenet_v2(num_classes=len(class_names), pretrained=False)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CheckpointIncompatibleError(
            "Checkpoint weights do not match MobileNetV2 with "
            f"{len(class_names)} output classes: {str(exc).splitlines()[0]}"
        ) from exc

    model.to(device or cfg.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return LoadedModel(
        model=model,
        class_names=class_names,
        image_size=image_size,
        mean=mean,
        std=std,
        checkpoint_path=path,
        checkpoint_sha256=file_sha256(path),
        metadata={k: v for k, v in meta.items() if isinstance(v, (int, float, str, list, dict))},
        notes=notes,
    )


def load_temperature(report_dir: Path, checkpoint_sha256: str) -> float | None:
    """Return the validation-fitted temperature only if it was fitted for *this* checkpoint."""
    path = Path(report_dir) / CALIBRATION_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("checkpoint_sha256") == checkpoint_sha256 and float(data["temperature"]) > 0:
            return float(data["temperature"])
    except (ValueError, KeyError, OSError):
        return None
    return None


class Predictor:
    """Runs preprocessing, inference, optional Grad-CAM and quality checks."""

    def __init__(self, loaded: LoadedModel, cfg: Config, temperature: float | None = None):
        self.loaded = loaded
        self.cfg = cfg
        self.temperature = temperature
        self.device = next(loaded.model.parameters()).device

    @property
    def class_names(self) -> list[str]:
        return self.loaded.class_names

    def _input(self, image: Image.Image) -> torch.Tensor:
        return to_model_input(image, self.loaded.image_size, self.loaded.mean, self.loaded.std, self.cfg.pixel_scale).to(self.device)

    @torch.inference_mode()
    def logits(self, image: Image.Image) -> np.ndarray:
        return self.loaded.model(self._input(image))[0].cpu().numpy()

    def predict(self, image: Image.Image, with_gradcam: bool = True) -> Prediction:
        x = self._input(image)
        with torch.inference_mode():
            logits = self.loaded.model(x)[0].double().cpu().numpy()

        t = self.temperature or 1.0
        z = logits / t
        z -= z.max()
        probs = np.exp(z) / np.exp(z).sum()

        order = np.argsort(-probs)
        names = self.loaded.class_names
        top = int(order[0])
        second = float(probs[order[1]]) if len(order) > 1 else 0.0
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum() / np.log(len(probs)))
        confidence = float(probs[top])

        result = Prediction(
            class_name=names[top],
            display_name=self.cfg.display_name(names[top]),
            class_index=top,
            confidence=confidence,
            ranked=[(names[i], float(probs[i])) for i in order[: self.cfg.top_k]],
            probability_kind="temperature-scaled softmax" if self.temperature else "raw softmax",
            margin=confidence - second,
            normalized_entropy=entropy,
            low_confidence=confidence < self.cfg.low_confidence_threshold,
            quality_warnings=check_quality(image),
        )

        if with_gradcam:
            try:
                with GradCAM(self.loaded.model) as cam_engine:
                    cam, _, _ = cam_engine(x, class_idx=top)
                result.cam = cam
                result.overlay = overlay_heatmap(image.resize(self.loaded.image_size, Image.Resampling.BILINEAR), cam)
            except Exception as exc:  # noqa: BLE001 - explanation is optional; never block the prediction
                result.gradcam_error = f"Grad-CAM unavailable: {exc}"
        return result

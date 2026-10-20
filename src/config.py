"""Project configuration: one place for paths, class mapping and preprocessing.

Everything is resolved relative to the project root, so the project runs on any
machine (no ``D:\\...`` or ``/content/...`` paths).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"

# Environment variables (see .env.example). They only override file paths.
ENV_MODEL_PATH = "BLOOD_CELL_MODEL_PATH"
ENV_CONFIG_PATH = "BLOOD_CELL_CONFIG"
ENV_DATA_ROOT = "BLOOD_CELL_DATA_ROOT"


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Resolve ``path`` against ``base`` unless it is already absolute."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    model_path: Path
    class_map_path: Path
    data_root: Path
    splits_csv: Path
    image_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    pixel_scale: float
    display_names: dict[str, str] = field(default_factory=dict)
    top_k: int = 8
    low_confidence_threshold: float = 0.70
    max_upload_mb: float = 10.0
    min_image_side_px: int = 64
    max_image_side_px: int = 8000
    allowed_extensions: tuple[str, ...] = ("png", "jpg", "jpeg")
    device: str = "cpu"
    seed: int = 42

    def display_name(self, class_name: str) -> str:
        return self.display_names.get(class_name, class_name.replace("_", " ").title())


def load_config(path: str | Path | None = None) -> Config:
    """Load ``configs/config.json`` (or ``$BLOOD_CELL_CONFIG``) plus env overrides."""
    cfg_path = Path(path) if path else Path(os.environ.get(ENV_CONFIG_PATH, DEFAULT_CONFIG_PATH))
    cfg_path = resolve_path(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    pre, inf = raw["preprocessing"], raw["inference"]
    model_path = os.environ.get(ENV_MODEL_PATH) or raw["model"]["checkpoint_path"]
    data_root = os.environ.get(ENV_DATA_ROOT) or raw["data"]["data_root"]
    return Config(
        raw=raw,
        model_path=resolve_path(model_path),
        class_map_path=resolve_path(raw["classes"]["class_to_idx_path"]),
        data_root=resolve_path(data_root),
        splits_csv=resolve_path(raw["data"]["splits_csv"]),
        image_size=tuple(pre["image_size"]),  # type: ignore[arg-type]
        mean=tuple(pre["mean"]),  # type: ignore[arg-type]
        std=tuple(pre["std"]),  # type: ignore[arg-type]
        pixel_scale=float(pre["pixel_scale"]),
        display_names=dict(raw["classes"].get("display_names", {})),
        top_k=int(inf["top_k"]),
        low_confidence_threshold=float(inf["low_confidence_threshold"]),
        max_upload_mb=float(inf["max_upload_mb"]),
        min_image_side_px=int(inf["min_image_side_px"]),
        max_image_side_px=int(inf["max_image_side_px"]),
        allowed_extensions=tuple(e.lower() for e in inf["allowed_extensions"]),
        device=str(inf.get("device", "cpu")),
        seed=int(raw["reproducibility"]["seed"]),
    )


def load_class_names(class_map_path: Path) -> list[str]:
    """Return class names ordered by integer label from ``class_to_idx.json``."""
    with Path(class_map_path).open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    return class_names_from_mapping(class_to_idx)


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("class_to_idx must be a non-empty mapping.")
    indices = sorted(class_to_idx.values())
    if indices != list(range(len(class_to_idx))):
        raise ValueError(f"class_to_idx labels must be 0..N-1 without gaps, got {indices}.")
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

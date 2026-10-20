"""Shared fixtures.

IMPORTANT: the checkpoint and images below are *synthetic test fixtures*
(random weights, random pixels). They verify code paths - loading, shapes,
error handling, pipeline plumbing - and say nothing about model accuracy.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config, load_config  # noqa: E402

CLASS_NAMES = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}


def png_bytes(size=(120, 120), color=(200, 120, 180), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def real_config() -> Config:
    return load_config()


@pytest.fixture()
def synthetic_assets(tmp_path, real_config) -> Config:
    """A tiny on-disk dataset + class map + manifest, wired into a Config."""
    rng = np.random.default_rng(0)
    data_root = tmp_path / "data" / "PBC"
    rows = []
    for cls in CLASS_NAMES:
        (data_root / cls).mkdir(parents=True)
        for k in range(6):
            arr = rng.integers(60, 255, size=(80, 80, 3), dtype=np.uint8)
            name = f"{cls}_{k}.jpg"
            Image.fromarray(arr).save(data_root / cls / name)
            rows.append(
                {
                    # Mimic the real manifest: absolute Windows path that does NOT exist here.
                    "path": f"D:\\blood-cell-capstone\\data\\raw\\PBC_dataset_normal_DIB\\{cls}\\{name}",
                    "rel_path": f"{cls}/{name}",
                    "class_name": cls,
                    "split": ["train", "train", "val", "val", "test", "test"][k],
                }
            )
    splits = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(splits, index=False)
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps(CLASS_TO_IDX), encoding="utf-8")
    return dataclasses.replace(
        real_config,
        data_root=data_root,
        splits_csv=splits,
        class_map_path=class_map,
        model_path=tmp_path / "models" / "ckpt.pt",
    )


@pytest.fixture()
def synthetic_checkpoint(synthetic_assets) -> Config:
    """Write a random-weight checkpoint in the same format the trainer saves."""
    import torch

    from src.models.mobilenet import build_mobilenet_v2

    torch.manual_seed(0)
    model = build_mobilenet_v2(8)
    path = synthetic_assets.model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": CLASS_TO_IDX,
            "image_size": [224, 224],
            "epoch": 1,
            "val_accuracy": 0.0,
            "val_macro_f1": 0.0,
            "model_name": "MobileNetV2",
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        path,
    )
    return synthetic_assets

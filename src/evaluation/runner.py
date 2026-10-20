"""Shared helper: run a checkpoint over a dataset split and collect logits."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.image_dataset import BloodCellDataset
from src.inference.predictor import LoadedModel


def make_dataset(split: str, cfg: Config, loaded: LoadedModel) -> BloodCellDataset:
    return BloodCellDataset(
        split,
        csv_path=cfg.splits_csv,
        image_size=loaded.image_size,
        class_map_path=cfg.class_map_path,
        data_root=cfg.data_root,
    )


@torch.inference_mode()
def collect_logits(loaded: LoadedModel, dataset, batch_size: int = 64, num_workers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(logits[N, C], labels[N])`` in dataset order. Dataset yields [0,1] tensors."""
    device = next(loaded.model.parameters()).device
    mean = torch.tensor(loaded.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(loaded.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = (images.to(device) - mean) / std
        all_logits.append(loaded.model(images).float().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)

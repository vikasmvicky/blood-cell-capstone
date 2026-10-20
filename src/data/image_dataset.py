
"""PyTorch dataset loader for blood-cell image classification."""

import random
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_CSV = PROJECT_ROOT / "outputs" / "reports" / "splits.csv"
CLASS_MAP_PATH = PROJECT_ROOT / "outputs" / "reports" / "class_to_idx.json"

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "PBC_dataset_normal_DIB"

IMAGE_SIZE = (224, 224)

VALID_SPLITS = {"train", "val", "test"}


def load_class_mapping(class_map_path=CLASS_MAP_PATH):
    """Load the saved mapping from class names to integer labels."""
    with Path(class_map_path).open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)

    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError(
            "class_to_idx.json must contain a non-empty mapping."
        )

    return class_to_idx


def resolve_image_path(row, data_root=DATA_ROOT):
    """Locate an image on this machine.

    The split manifest stores the absolute path from the machine that built it
    (e.g. ``D:\\blood-cell-capstone\\data\\raw\\...``). That path usually does
    not exist elsewhere, so fall back to ``data_root / rel_path`` and then to
    ``data_root / <class_name> / <filename>``.
    """
    data_root = Path(data_root)
    candidates = []

    recorded = row.get("path") if hasattr(row, "get") else None
    if isinstance(recorded, str) and recorded:
        candidates.append(Path(recorded))

    rel_path = row.get("rel_path") if hasattr(row, "get") else None
    if isinstance(rel_path, str) and rel_path:
        candidates.append(data_root / Path(*PureWindowsPath(rel_path).parts))

    if isinstance(recorded, str) and recorded:
        filename = PureWindowsPath(recorded).name
        candidates.append(data_root / str(row["class_name"]) / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(c) for c in candidates) or "<no path columns>"
    raise FileNotFoundError(
        f"Image not found. Tried: {tried}. Place the PBC dataset under "
        f"'{data_root}' (see ASSETS.md)."
    )


class BloodCellDataset(Dataset):
    """Load blood-cell images one at a time instead of all into RAM."""

    def __init__(
        self,
        split,
        csv_path=SPLIT_CSV,
        image_size=IMAGE_SIZE,
        class_map_path=CLASS_MAP_PATH,
        data_root=DATA_ROOT,
    ):
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Choose from {sorted(VALID_SPLITS)}."
            )

        self.csv_path = Path(csv_path)
        self.image_size = tuple(image_size)
        self.class_to_idx = load_class_mapping(class_map_path)
        self.data_root = Path(data_root)

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Split manifest not found: {self.csv_path}. Create it with "
                f"'python -m src.data.inspect_dataset' and "
                f"'python -m src.data.split_dataset --seed 123' (see ASSETS.md)."
            )

        df = pd.read_csv(self.csv_path)

        required_columns = {"path", "class_name", "split"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        self.df = df.loc[
            df["split"] == split
        ].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(
                f"No images found for split '{split}'."
            )

        unknown_classes = (
            set(self.df["class_name"]) - set(self.class_to_idx)
        )

        if unknown_classes:
            raise ValueError(
                f"Classes missing from class mapping: "
                f"{sorted(unknown_classes)}"
            )

        self.split = split

    def augment_image(self, image):
        """Apply mild random transformations to training images only."""

        if random.random() < 0.5:
            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        if random.random() < 0.5:
            image = image.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )

        angle = random.uniform(-15, 15)

        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
        )

        return image

    def __len__(self):
        """Return the number of images in this split."""
        return len(self.df)

    def __getitem__(self, index):
        """Return one image tensor and its integer class label."""

        row = self.df.iloc[index]
        image_path = resolve_image_path(row, self.data_root)

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.split == "train":
                image = self.augment_image(image)

            image = image.resize(
                self.image_size,
                Image.Resampling.BILINEAR,
            )

            image_array = (
                np.asarray(image, dtype=np.float32) / 255.0
            )

        # Convert H x W x C into C x H x W for PyTorch.
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(
                image_array.transpose(2, 0, 1)
            )
        )

        label = self.class_to_idx[row["class_name"]]
        label_tensor = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image_tensor, label_tensor

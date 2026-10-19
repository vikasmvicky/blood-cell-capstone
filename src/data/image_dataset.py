
"""PyTorch dataset loader for blood-cell image classification."""

import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_CSV = PROJECT_ROOT / "outputs" / "reports" / "splits.csv"
CLASS_MAP_PATH = PROJECT_ROOT / "outputs" / "reports" / "class_to_idx.json"

IMAGE_SIZE = (224, 224)

VALID_SPLITS = {"train", "val", "test"}


def load_class_mapping():
    """Load the saved mapping from class names to integer labels."""
    with CLASS_MAP_PATH.open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)

    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError(
            "class_to_idx.json must contain a non-empty mapping."
        )

    return class_to_idx


class BloodCellDataset(Dataset):
    """Load blood-cell images one at a time instead of all into RAM."""

    def __init__(
        self,
        split,
        csv_path=SPLIT_CSV,
        image_size=IMAGE_SIZE,
    ):
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Choose from {sorted(VALID_SPLITS)}."
            )

        self.csv_path = Path(csv_path)
        self.image_size = tuple(image_size)
        self.class_to_idx = load_class_mapping()

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
        image_path = Path(row["path"])

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

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

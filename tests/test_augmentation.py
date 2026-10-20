
from unittest.mock import patch

import pytest
from PIL import Image

from src.data.image_dataset import BloodCellDataset


def main():
    train_ds = BloodCellDataset("train")
    val_ds = BloodCellDataset("val")
    test_ds = BloodCellDataset("test")

    sample = Image.new("RGB", (100, 100), color="white")

    # Training augmentation should call the augmentation method.
    with patch.object(
        train_ds,
        "augment_image",
        wraps=train_ds.augment_image,
    ) as mocked:
        train_ds[0]
        assert mocked.call_count == 1

    # Validation and test should not call augmentation.
    with patch.object(
        val_ds,
        "augment_image",
        wraps=val_ds.augment_image,
    ) as mocked:
        val_ds[0]
        assert mocked.call_count == 0

    with patch.object(
        test_ds,
        "augment_image",
        wraps=test_ds.augment_image,
    ) as mocked:
        test_ds[0]
        assert mocked.call_count == 0

    # Confirm the augmentation method returns a PIL image.
    augmented = train_ds.augment_image(sample)
    assert isinstance(augmented, Image.Image)

    print("Training augmentation: PASS")
    print("Validation augmentation disabled: PASS")
    print("Test augmentation disabled: PASS")
    print("Augmented image type: PASS")
    print("SUCCESS: All augmentation checks passed.")


def test_augmentation_on_real_splits():
    """Needs the real split manifest and PBC images (see ASSETS.md); skipped otherwise."""
    from src.data.image_dataset import SPLIT_CSV

    if not SPLIT_CSV.is_file():
        pytest.skip(f"split manifest missing: {SPLIT_CSV}")
    try:
        main()
    except FileNotFoundError as exc:
        pytest.skip(f"dataset images missing: {exc}")


if __name__ == "__main__":
    main()

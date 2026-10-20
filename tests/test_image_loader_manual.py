
from torch.utils.data import DataLoader
from src.data.image_dataset import BloodCellDataset

import pytest


def main():
    expected_counts = {
        "train": 11964,
        "val": 2564,
        "test": 2562,
    }

    for split, expected in expected_counts.items():
        dataset = BloodCellDataset(split)

        print(f"\n--- {split.upper()} ---")
        print(f"Images: {len(dataset)}")
        print(f"Expected count: {expected}")
        print(f"Count correct: {len(dataset) == expected}")

        image, label = dataset[0]
        print(f"Individual image shape: {tuple(image.shape)}")
        print(f"First image label: {label.item()}")

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        images, labels = next(iter(loader))

        print(f"Batch image shape: {tuple(images.shape)}")
        print(f"Batch label shape: {tuple(labels.shape)}")

        assert len(dataset) == expected
        assert tuple(image.shape) == (3, 224, 224)
        assert tuple(images.shape) == (4, 3, 224, 224)
        assert tuple(labels.shape) == (4,)

    print("\nSUCCESS: All splits and batch checks passed.")


def test_real_split_sizes_and_batches():
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

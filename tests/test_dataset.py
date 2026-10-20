import pytest
import torch
from PIL import Image

from src.data.image_dataset import BloodCellDataset, resolve_image_path
from tests.conftest import CLASS_NAMES, CLASS_TO_IDX


def make(split, cfg):
    return BloodCellDataset(split, csv_path=cfg.splits_csv, class_map_path=cfg.class_map_path, data_root=cfg.data_root)


def test_split_sizes_and_class_mapping(synthetic_assets):
    for split in ("train", "val", "test"):
        ds = make(split, synthetic_assets)
        assert len(ds) == 2 * len(CLASS_NAMES)
        assert ds.class_to_idx == CLASS_TO_IDX


def test_item_shape_dtype_and_label(synthetic_assets):
    ds = make("test", synthetic_assets)
    image, label = ds[0]
    assert tuple(image.shape) == (3, 224, 224)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert label.item() == CLASS_TO_IDX[ds.df.loc[0, "class_name"]]


def test_windows_absolute_path_falls_back_to_data_root(synthetic_assets):
    ds = make("val", synthetic_assets)
    row = ds.df.iloc[0]
    assert row["path"].startswith("D:\\")          # recorded path does not exist here
    assert resolve_image_path(row, synthetic_assets.data_root).is_file()


def test_missing_image_gives_helpful_error(synthetic_assets, tmp_path):
    ds = make("val", synthetic_assets)
    ds.data_root = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError, match="ASSETS.md"):
        ds[0]


def test_invalid_split_and_missing_manifest(synthetic_assets, tmp_path):
    with pytest.raises(ValueError):
        BloodCellDataset("holdout", csv_path=synthetic_assets.splits_csv, class_map_path=synthetic_assets.class_map_path)
    with pytest.raises(FileNotFoundError, match="split_dataset"):
        BloodCellDataset("train", csv_path=tmp_path / "missing.csv", class_map_path=synthetic_assets.class_map_path)


def test_augmentation_only_on_train(synthetic_assets):
    train, val = make("train", synthetic_assets), make("val", synthetic_assets)
    assert isinstance(train.augment_image(Image.new("RGB", (50, 50))), Image.Image)
    calls = {"n": 0}
    orig = train.augment_image
    train.augment_image = lambda img: (calls.__setitem__("n", calls["n"] + 1), orig(img))[1]
    train[0]
    assert calls["n"] == 1
    val.augment_image = lambda img: pytest.fail("augmentation must not run on val")
    val[0]

import json

import numpy as np
import pytest
import torch

from src.inference.errors import CheckpointIncompatibleError, CheckpointNotFoundError, InvalidImageError
from src.inference.predictor import Predictor, load_checkpoint, load_temperature
from src.inference.service import predict_upload, validate_upload
from src.models.cnn_model import BloodCellCNN
from src.models.mobilenet import build_mobilenet_v2, count_parameters
from tests.conftest import CLASS_NAMES, png_bytes


def test_mobilenet_architecture_and_param_count():
    model = build_mobilenet_v2(8).eval()
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 8)
    assert 2.2e6 < count_parameters(model) < 2.25e6   # ~2.23M (matches the documented figure)


def test_cnn_baseline_forward():
    out = BloodCellCNN(8).eval()(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 8) and torch.isfinite(out).all()


def test_missing_checkpoint_raises_clear_error(synthetic_assets):
    with pytest.raises(CheckpointNotFoundError, match="ASSETS.md"):
        load_checkpoint(synthetic_assets)


def test_checkpoint_loads_and_predicts(synthetic_checkpoint):
    loaded = load_checkpoint(synthetic_checkpoint)
    assert loaded.class_names == CLASS_NAMES and loaded.image_size == (224, 224)
    assert not loaded.model.training
    image, pred = predict_upload(png_bytes(size=(150, 150)), "x.png", Predictor(loaded, synthetic_checkpoint), synthetic_checkpoint)
    assert pred.class_name in CLASS_NAMES
    assert len(pred.ranked) == 8
    probs = [p for _, p in pred.ranked]
    assert probs == sorted(probs, reverse=True) and abs(sum(probs) - 1) < 1e-6
    assert pred.confidence == probs[0] and 0 <= pred.normalized_entropy <= 1
    assert pred.probability_kind == "raw softmax"


def test_gradcam_output(synthetic_checkpoint):
    predictor = Predictor(load_checkpoint(synthetic_checkpoint), synthetic_checkpoint)
    _, pred = predict_upload(png_bytes(size=(150, 150), color=(180, 90, 160)), "x.png", predictor, synthetic_checkpoint, with_gradcam=True)
    assert pred.gradcam_error is None
    assert pred.cam.shape == (224, 224) and 0.0 <= pred.cam.min() and pred.cam.max() <= 1.0
    assert pred.overlay.size == (224, 224)


def test_gradcam_is_deterministic_and_leaves_model_unchanged(synthetic_checkpoint):
    loaded = load_checkpoint(synthetic_checkpoint)
    predictor = Predictor(loaded, synthetic_checkpoint)
    from PIL import Image

    img = Image.new("RGB", (100, 100), (150, 60, 140))
    before = {k: v.clone() for k, v in loaded.model.state_dict().items()}
    a = predictor.predict(img).cam
    b = predictor.predict(img).cam
    assert np.allclose(a, b)
    assert all(torch.equal(before[k], v) for k, v in loaded.model.state_dict().items())


def test_incompatible_checkpoints_rejected(synthetic_checkpoint, tmp_path):
    cfg = synthetic_checkpoint
    # wrong number of classes in the head
    bad = build_mobilenet_v2(5).state_dict()
    torch.save({"model_state_dict": bad, "class_to_idx": {n: i for i, n in enumerate(CLASS_NAMES)}}, cfg.model_path)
    with pytest.raises(CheckpointIncompatibleError, match="do not match"):
        load_checkpoint(cfg)
    # class order disagrees with class_to_idx.json
    swapped = dict(zip(reversed(CLASS_NAMES), range(8)))
    torch.save({"model_state_dict": build_mobilenet_v2(8).state_dict(), "class_to_idx": swapped}, cfg.model_path)
    with pytest.raises(CheckpointIncompatibleError, match="differs"):
        load_checkpoint(cfg)
    # garbage file
    cfg.model_path.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointIncompatibleError, match="Could not read"):
        load_checkpoint(cfg)


def test_service_rejects_bad_uploads(synthetic_checkpoint):
    with pytest.raises(InvalidImageError, match="Unsupported file type"):
        validate_upload(png_bytes(), "cell.gif", synthetic_checkpoint)
    with pytest.raises(InvalidImageError):
        validate_upload(b"garbage", "cell.png", synthetic_checkpoint)


def test_temperature_only_applies_to_matching_checkpoint(tmp_path):
    (tmp_path / "calibration.json").write_text(json.dumps({"temperature": 1.7, "checkpoint_sha256": "abc"}))
    assert load_temperature(tmp_path, "abc") == 1.7
    assert load_temperature(tmp_path, "different") is None
    assert load_temperature(tmp_path / "none", "abc") is None

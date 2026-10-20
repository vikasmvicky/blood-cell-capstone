"""Unit tests for the Grad-CAM faithfulness experiment (synthetic model, no real assets)."""

import json

import numpy as np
import pytest
import torch

from src.experiments import gradcam_faithfulness as F


def test_top_fraction_mask_counts():
    cam = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = F.top_fraction_mask(cam, 0.10)
    assert mask.sum() == 10
    assert set(np.argwhere(mask)[:, 0]) >= {9}  # highest values are kept
    assert F.top_fraction_mask(cam, 0.5).sum() == 50
    assert F.top_fraction_mask(cam, 0.001).sum() == 1  # at least one pixel


def test_mean_colour_baseline_and_perturb_modes():
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    arr[..., 0] = 200.0  # mean colour = (200, 0, 0)
    base = F.mean_colour_baseline(arr)
    assert np.allclose(base, np.array([200.0, 0.0, 0.0]))
    order = np.arange(16)

    deleted = F.perturb(arr, order, k=4, mode="deletion")
    assert np.allclose(deleted.reshape(-1, 3)[:4], [200.0, 0.0, 0.0])   # replaced by mean colour
    assert np.allclose(deleted.reshape(-1, 3)[4:, 0], 200.0)            # rest untouched

    inserted = F.perturb(arr, order, k=4, mode="insertion")
    assert np.allclose(inserted.reshape(-1, 3)[:4], [200.0, 0.0, 0.0])  # original pixels back
    # rest is the mean-colour baseline; use an image whose mean differs from its pixels
    arr2 = np.zeros((4, 4, 3), dtype=np.float32)
    arr2[..., 0] = (np.arange(16, dtype=np.float32) * 10).reshape(4, 4)  # R varies 0..150
    arr2[..., 1] = 50.0
    arr2[..., 2] = 10.0                                                   # mean colour = (75, 50, 10)
    ins2 = F.perturb(arr2, order, k=4, mode="insertion")
    assert np.allclose(ins2.reshape(-1, 3)[:4], arr2.reshape(-1, 3)[:4])
    assert np.allclose(ins2.reshape(-1, 3)[4:], [75.0, 50.0, 10.0])

    assert np.allclose(F.perturb(arr, order, 0, "deletion"), arr)                     # nothing deleted yet
    assert np.allclose(F.perturb(arr, order, 0, "insertion"), F.mean_colour_baseline(arr))
    assert np.allclose(F.perturb(arr, order, 16, "deletion"), F.mean_colour_baseline(arr))  # everything deleted
    assert np.allclose(F.perturb(arr, order, 16, "insertion"), arr)
    with pytest.raises(ValueError):
        F.perturb(arr, order, 4, mode="replace")


def test_curve_auc_known_values():
    x = np.array([0.0, 0.5, 1.0])
    assert F.curve_auc(np.array([1.0, 1.0, 1.0]), x) == pytest.approx(1.0)
    assert F.curve_auc(np.array([0.0, 0.0, 0.0]), x) == pytest.approx(0.0)
    assert F.curve_auc(np.array([1.0, 0.0, 1.0]), x) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        F.curve_auc(np.array([1.0]), x)


class _TinyModel(torch.nn.Module):
    """Fixed random linear map 4x4x3 -> 3 logits; different pixels get different weights."""

    def __init__(self, n_classes=3):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.register_buffer("w", torch.randn(4 * 4 * 3, n_classes, generator=g))

    def forward(self, x):
        return x.flatten(1) @ self.w


def test_run_image_curves_structure_and_endpoints():
    model = _TinyModel().eval()
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    cam = rng.uniform(0, 1, size=(4, 4)).astype(np.float32)
    fractions = np.linspace(0.0, 1.0, 5)
    out = F.run_image_curves(arr, cam, cls=1, model=model, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                             fractions=fractions, rngs=[np.random.default_rng(1), np.random.default_rng(2)])
    assert set(out) == {"gradcam", "random"}
    assert set(out["random"]) == {0, 1}
    for mode in ("deletion", "insertion"):
        curve = out["gradcam"][mode]
        assert curve.shape == (len(fractions),)
        assert np.all((curve >= 0) & (curve <= 1))
    # Deletion at 0% = original image; insertion at 0% = blank mean image (identical for random too)
    x0 = torch.from_numpy(F._to_norm_tensor(arr[None], (0.5,) * 3, (0.5,) * 3))
    p_orig = torch.softmax(model(x0), dim=1)[0, 1].item()
    assert out["gradcam"]["deletion"][0] == pytest.approx(p_orig, abs=1e-5)
    assert out["gradcam"]["deletion"][-1] == pytest.approx(out["random"][0]["deletion"][-1], abs=1e-5)
    # At 100% insertion every ordering gives back the original image
    assert out["gradcam"]["insertion"][-1] == pytest.approx(p_orig, abs=1e-5)
    assert out["random"][1]["insertion"][-1] == pytest.approx(p_orig, abs=1e-5)


def test_aggregate_results_and_plot(tmp_path):
    fractions = np.linspace(0.0, 1.0, 4)
    rng = np.random.default_rng(0)
    curves = []
    for _ in range(5):
        c = {"gradcam": {"deletion": rng.uniform(size=4), "insertion": rng.uniform(size=4)},
             "random": {i: {"deletion": rng.uniform(size=4), "insertion": rng.uniform(size=4)} for i in range(2)}}
        curves.append(c)
    agg = F.aggregate_results(curves, fractions)
    assert len(agg["curves"]["gradcam_deletion"]["mean"]) == 4
    for key in ("gradcam_deletion", "gradcam_insertion", "random_deletion", "random_insertion"):
        assert 0.0 <= agg["auc"][key]["mean"] <= 1.0
    out = tmp_path / "curves.png"
    F.plot_curves(agg, out)
    assert out.stat().st_size > 1000


def test_cli_end_to_end_synthetic(tmp_path, synthetic_checkpoint, monkeypatch):
    """Run the real CLI on a synthetic checkpoint + tiny synthetic dataset."""
    from src.experiments import gradcam_faithfulness as mod

    cfg = synthetic_checkpoint
    monkeypatch.setattr(mod, "load_config", lambda path=None: cfg)
    out_json = tmp_path / "faith.json"
    out_png = tmp_path / "faith.png"
    rc = mod.main(["--split", "val", "--max-images", "3", "--steps", "3", "--random-seeds", "2",
                   "--out", str(out_json), "--plot", str(out_png)])
    assert rc == 0
    report = json.loads(out_json.read_text())
    assert report["n_images"] == 3 and report["split"] == "val"
    assert report["random_baseline_seeds"] == 2
    assert len(report["curves"]["gradcam_deletion"]["mean"]) == 4  # steps + 1
    assert len(report["auc"]["gradcam_deletion"]["values"]) == 3
    assert len(report["checkpoint_sha256"]) == 64
    assert isinstance(report["summary"]["deletion_more_faithful_than_random"], bool)
    assert out_png.stat().st_size > 1000

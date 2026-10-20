"""Grad-CAM for the MobileNetV2 classifier (no extra dependencies).

Reference method: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (ICCV 2017).

For MobileNetV2 at 224x224 the target layer ``features[-1]`` has a 7x7 spatial
grid, so heat-maps are coarse (each cell covers 32x32 input pixels) and are
bilinearly upsampled. A heat-map shows where the network's evidence for the
selected class came from; it is a qualitative aid, not proof of correct
reasoning and not a segmentation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    """Hook-based Grad-CAM. Use as a context manager or call ``close()``."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inputs, output):
        self._activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._gradients = grad

    def close(self) -> None:
        self._fwd_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, torch.Tensor]:
        """Return ``(cam, class_idx, logits)``.

        ``cam`` is an ``HxW`` float32 array in [0, 1] at the input resolution.
        If ``class_idx`` is None the predicted (arg-max) class is explained.
        """
        was_training = self.model.training
        self.model.eval()
        self._activations = self._gradients = None
        try:
            with torch.enable_grad():
                x = x.detach().clone().requires_grad_(True)
                logits = self.model(x)
                if class_idx is None:
                    class_idx = int(logits.argmax(dim=1).item())
                self.model.zero_grad(set_to_none=True)
                logits[0, class_idx].backward()
        finally:
            self.model.train(was_training)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        acts = self._activations.detach()[0]            # (C, h, w)
        grads = self._gradients.detach()[0]             # (C, h, w)
        weights = grads.mean(dim=(1, 2), keepdim=True)  # global-average-pooled gradients
        cam = F.relu((weights * acts).sum(dim=0))       # (h, w)
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam_np = cam.cpu().numpy().astype(np.float32)
        peak = float(cam_np.max())
        cam_np = cam_np / peak if peak > 0 else np.zeros_like(cam_np)
        return cam_np, class_idx, logits.detach()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [0,1] heat-map (jet colormap) over ``image`` (resized to the cam size)."""
    h, w = cam.shape
    base = image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * heat.astype(np.float32)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

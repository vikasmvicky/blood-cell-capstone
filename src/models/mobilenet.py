"""MobileNetV2 factory shared by inference, evaluation and benchmarking.

The training scripts (train_mobilenet*.py) build the same architecture inline:
``torchvision.models.mobilenet_v2`` with ``classifier[1]`` replaced by
``Linear(last_channel, num_classes)``. Keeping one factory here guarantees that
a checkpoint's state-dict keys always match what inference constructs.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_mobilenet_v2(num_classes: int = 8, pretrained: bool = False) -> nn.Module:
    """Return MobileNetV2 with an ``num_classes``-way classifier head."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

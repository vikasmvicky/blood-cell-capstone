
"""CNN model for eight-class blood-cell image classification."""

import torch
import torch.nn as nn


class BloodCellCNN(nn.Module):
    """A compact CNN built from scratch for blood-cell classification."""

    def __init__(self, num_classes=8):
        super().__init__()

        # Feature extraction: learn patterns from image pixels.
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Reduce the spatial dimensions to 1 x 1 per feature channel.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification: map learned features to eight class scores.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """Return raw scores (logits) for each blood-cell class."""
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn

from src.models.cnn_model import BloodCellCNN


def main():
    torch.manual_seed(42)

    model = BloodCellCNN(num_classes=8)
    model.train()

    # Simulate a batch of two RGB images.
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1], dtype=torch.long)

    # Forward pass.
    logits = model(images)

    assert logits.shape == (2, 8)
    assert torch.isfinite(logits).all()

    print("Forward pass: PASS")
    print("Input shape:", tuple(images.shape))
    print("Output shape:", tuple(logits.shape))

    # Calculate loss and perform backpropagation.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    print("Training step: PASS")
    print("Loss:", round(loss.item(), 4))
    print("Backpropagation: PASS")
    print("Optimizer update: PASS")

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print("Trainable parameters:", parameter_count)
    print("SUCCESS: CNN smoke test passed.")


if __name__ == "__main__":
    main()

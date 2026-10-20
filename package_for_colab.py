
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "blood_cell_colab_package.zip"

required_files = [
    "outputs/reports/splits.csv",
    "outputs/reports/class_to_idx.json",
    "src/data/image_dataset.py",
    "src/models/cnn_model.py",
]

dataset_dir = ROOT / "data/raw/PBC_dataset_normal_DIB"

if not dataset_dir.is_dir():
    raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

for relative_path in required_files:
    if not (ROOT / relative_path).is_file():
        raise FileNotFoundError(f"Required file missing: {relative_path}")

with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
    for relative_path in required_files:
        archive.write(
            ROOT / relative_path,
            arcname=relative_path,
        )

    image_count = 0
    for image_path in dataset_dir.rglob("*"):
        if image_path.is_file():
            archive.write(
                image_path,
                arcname=image_path.relative_to(ROOT).as_posix(),
            )
            image_count += 1

print("Package created:", OUTPUT)
print("Dataset files packaged:", image_count)
print("ZIP size (MB):", round(OUTPUT.stat().st_size / (1024 * 1024), 2))

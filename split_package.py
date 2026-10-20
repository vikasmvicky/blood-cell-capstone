from pathlib import Path

source = Path("blood_cell_colab_package.zip")
chunk_size = 40 * 1024 * 1024

if not source.exists():
    raise FileNotFoundError(f"Missing file: {source}")

with source.open("rb") as f:
    index = 1
    while True:
        data = f.read(chunk_size)
        if not data:
            break
        part = Path(f"blood_cell_part_{index:02d}.bin")
        part.write_bytes(data)
        print(f"Created {part}: {len(data) / (1024 * 1024):.1f} MB")
        index += 1

print(f"Done: {index - 1} parts created.")

from __future__ import annotations
import argparse
import hashlib
import os
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HINT_WORDS = ("mask", "seg", "annot", "label", "bbox", "boxes")
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def scan_tree(data_dir:Path)->list[dict]:
    records=[]
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        class_name=path.parent.name
        records.append({"path":str(path.resolve()),"filename":path.name,"class_name":class_name,"extension":path.suffix.lower(),"size_bytes":path.stat().st_size,})
    return records
def inspect_image(record:dict)->dict:
    result=record.copy()
    try:
        with Image.open(record["path"]) as image:
            result["width"]=image.width
            result["height"]=image.height
            result["image_format"]=image.format
            result["mode"]=image.mode
            image.verify()
            result["readable"]=True
            result["error"]=""
    except Exception as exc:
        result["width"]=None
        result["height"]=None
        result["image_format"]=None
        result["mode"]=None
        result["image_format"]=None
        result["mode"]=None
        result["readable"]=False
        result["error"]=str(exc)
    return result
def calculate_sha256(file_path:str)->str:
    sha256=hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024*1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()   
def find_extract_duplicates(records:list[dict])->dict:
    hash_groups={}
    for record in records:
        try:
            file_hash = record.get("sha256") or calculate_sha256(record["path"])
        except OSError as exc:
            print(f"could not hash{record['path']}:{exc}")
            continue
        hash_groups.setdefault(file_hash,[]).append(record["path"])
        duplicates={file_hash:paths for file_hash,paths in hash_groups.items() if len(paths)>1}
    return duplicates
def summarize_classes(records:list[dict])->dict:
    class_counts=Counter(record["class_name"] for record in records)
    summary={"total_images":len(records),"total_classes":len(class_counts),"class_counts":dict(sorted(class_counts.items()))}
    return summary
def save_csv(records:list[dict],output_path:Path)->None:
    output_path.parent.mkdir(parents=True,exist_ok=True)
    dataframe=pd.DataFrame(records)
    dataframe.to_csv(output_path,index=False)
    print(f"Saved Csv report:{output_path}")
def save_json(data:dict,output_path:Path)->None:
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with open(output_path, "w",encoding="utf-8") as file:
        json.dump(data, file,indent=2,default=str)

    print(f"Saved Json report:{output_path}")

def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description="Inspect a blood-cell image dataset."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing the dataset.",
    )

    parser.add_argument(
        "--near-dup",
        action="store_true",
        help="Enable near-duplicate detection.",
    )

    return parser.parse_args()

def main() -> None:
    """Run the dataset inspection workflow."""
    args = parse_args()
    data_dir = args.data_dir.resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {data_dir}")

    print(f"Inspecting dataset: {data_dir}")

    records = scan_tree(data_dir)
    print(f"Image files discovered: {len(records)}")

    if not records:
        raise SystemExit("No supported image files were found.")

    inspected_records = [
        inspect_image(record) for record in records
    ]

    for record in inspected_records:
        record["rel_path"] = Path(record["path"]).relative_to(
            data_dir
        ).as_posix()

        record["class"] = record["class_name"]

        try:
            record["sha256"] = calculate_sha256(record["path"])
        except OSError as exc:
            record["sha256"] = None
            print(f"Could not hash {record['path']}: {exc}")

    summary = summarize_classes(inspected_records)

    readable_count = sum(
        record["readable"] for record in inspected_records
    )
    unreadable_count = len(inspected_records) - readable_count

    summary["readable_images"] = readable_count
    summary["unreadable_images"] = unreadable_count

    print(f"Readable images: {readable_count}")
    print(f"Unreadable images: {unreadable_count}")
    print("\nClass counts:")

    for class_name, count in summary["class_counts"].items():
        print(f"  {class_name}: {count}")

    save_csv(
        inspected_records,
        REPORT_DIR / "image_manifest.csv",
    )

    save_json(
        summary,
        REPORT_DIR / "dataset_summary.json",
    )

    duplicates = find_extract_duplicates(inspected_records)

    duplicate_summary = {
        "duplicate_groups": len(duplicates),
        "duplicate_files": sum(
            len(paths) for paths in duplicates.values()
        ),
        "groups": duplicates,
    }

    save_json(
        duplicate_summary,
        REPORT_DIR / "exact_duplicates.json",
    )


if __name__ == "__main__":
    main()

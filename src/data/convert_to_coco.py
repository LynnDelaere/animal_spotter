"""Convert Open Images dataset to COCO JSON format.

Robust alternative to Hugging Face Datasets to avoid Segmentation Faults.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]
from PIL import Image
from tqdm import tqdm

# Root directory setup
ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(path: str) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def get_class_mapping(
    metadata_dir: Path, target_classes: list[str]
) -> tuple[dict[str, str], dict[str, int]]:
    """Creates mapping from OpenImages LabelName to ID and readable name."""
    desc_path = metadata_dir / "oidv7-class-descriptions-boxable.csv"
    if not desc_path.exists():
        raise FileNotFoundError(f"Metadata missing: {desc_path}")

    # Load class descriptions
    df = pd.read_csv(desc_path, header=None, names=["LabelName", "DisplayName"])

    # Filter on our target classes
    filtered = df[df["DisplayName"].isin(target_classes)]

    # Check for missing classes
    found = set(filtered["DisplayName"].unique())
    missing = set(target_classes) - found
    if missing:
        print(f"WARNING: Classes not found in metadata: {missing}")

    # Create mappings
    # MID (e.g., /m/011k07) -> Readable (e.g., 'Cat')
    mid_to_name = pd.Series(
        filtered.DisplayName.values, index=filtered.LabelName
    ).to_dict()

    # Readable -> COCO Category ID (1, 2, 3...)
    # Note: COCO usually starts at 1, but 0 is also fine depending on framework.
    # Let's use 0-indexed for simplicity with PyTorch mapping later.
    name_to_id = {name: idx for idx, name in enumerate(target_classes)}

    return mid_to_name, name_to_id


def create_coco_json(
    split: str,
    metadata_dir: Path,
    images_dir: Path,
    output_path: Path,
    target_classes: list[str],
) -> None:
    """Convert Open Images data to COCO JSON format for a given split."""
    print(f"\nProcessing split: {split} -> {output_path}")

    mid_to_name, name_to_id = get_class_mapping(metadata_dir, target_classes)
    target_mids = set(mid_to_name.keys())

    # 1. Setup COCO dictionary structure
    coco_output: dict[str, Any] = {
        "info": {"description": f"Open Images subset: {split}", "year": 2024},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx, "name": name, "supercategory": "animal"}
            for name, idx in name_to_id.items()
        ],
    }

    # 2. Get local images and get their dimensions
    split_img_dir = images_dir / split
    if not split_img_dir.exists():
        print(f"Skipping {split} (folder not found)")
        return

    image_files = sorted(list(split_img_dir.glob("*.jpg")))
    if not image_files:
        print(f"Skipping {split} (no images found)")
        return

    print(f"  Indexing {len(image_files)} images...")

    # Map filename (without ext) to image_id (integer) for COCO
    # We use a simple counter for Image IDs
    filename_to_id = {}

    # We also need to map 'Original ImageID string' (from CSV) to our new Int ID
    str_id_to_int_id = {}

    # Also store dimensions for coordinate conversion
    image_dims = {}  # int_id -> (width, height)

    for idx, img_path in enumerate(tqdm(image_files, desc="Reading Image Dims")):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"  Error reading {img_path}: {e}. Skipping.")
            continue

        str_id = img_path.stem  # e.g. "0a1b2c..."
        int_id = idx  # 0, 1, 2...

        filename_to_id[str_id] = int_id
        str_id_to_int_id[str_id] = int_id
        image_dims[int_id] = (width, height)

        coco_output["images"].append(
            {
                "id": int_id,
                "file_name": img_path.name,  # relative to image dir
                "width": width,
                "height": height,
            }
        )

    # 3. Process Annotations
    annot_csv = metadata_dir / f"{split}-annotations-bbox.csv"
    print(f"  Processing annotations from {annot_csv.name}...")

    annot_id_counter = 0

    # Pandas chunked reading with explicit dtypes and engine to avoid pyarrow issues
    chunk_size = 100_000
    cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]

    for chunk in pd.read_csv(
        annot_csv,
        usecols=cols,
        chunksize=chunk_size,
        engine="c",
        dtype={
            "ImageID": "string",
            "LabelName": "string",
            "XMin": "float64",
            "XMax": "float64",
            "YMin": "float64",
            "YMax": "float64",
        },
        memory_map=False,
        low_memory=False,
    ):
        filtered = chunk[chunk["LabelName"].isin(target_mids)]
        filtered = filtered[filtered["ImageID"].isin(str_id_to_int_id.keys())]

        for _, row in filtered.iterrows():
            str_img_id = row["ImageID"]
            int_img_id = str_id_to_int_id[str_img_id]
            width, height = image_dims[int_img_id]

            x_min = row["XMin"] * width
            x_max = row["XMax"] * width
            y_min = row["YMin"] * height
            y_max = row["YMax"] * height

            w_box = x_max - x_min
            h_box = y_max - y_min

            class_name = mid_to_name[row["LabelName"]]
            category_id = name_to_id[class_name]

            ann = {
                "id": annot_id_counter,
                "image_id": int_img_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, w_box, h_box],
                "area": w_box * h_box,
                "iscrowd": 0,
            }
            coco_output["annotations"].append(ann)
            annot_id_counter += 1

    # 4. Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            coco_output, f
        )  # No indent to save space, or indent=2 for readability

    print(f"  Saved {len(coco_output['annotations'])} annotations to {output_path}")


def main() -> None:
    """Entry point for converting Open Images to COCO format."""
    parser = argparse.ArgumentParser(description="Convert Open Images to COCO JSON")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Parse paths from config
    version = cfg["dataset"]["version"]
    classes = cfg["dataset"]["classes"]

    # Raw Dir (Input)
    raw_dir_str = cfg["paths"]["raw_dir"]
    raw_base = (ROOT_DIR / raw_dir_str).resolve()
    # Handle versioned structure if present
    # Check if data/openimages_animals/v1 exists, else use base
    raw_dir = raw_base / version if (raw_base / version).exists() else raw_base

    # Processed Dir (Output)
    processed_dir_str = cfg["paths"]["processed_dir"]
    output_dir = (ROOT_DIR / processed_dir_str / version).resolve()

    print(f"Converting Data Version: {version}")
    print(f"Input: {raw_dir}")
    print(f"Output: {output_dir}")

    metadata_dir = raw_dir / "metadata"
    if not metadata_dir.exists():
        # Fallback for structure: raw/v1/metadata vs raw/metadata
        metadata_dir = raw_base / "metadata"

    images_dir = raw_dir / "images"

    for split in ["train", "validation", "test"]:
        json_path = output_dir / f"{split}.json"
        create_coco_json(split, metadata_dir, images_dir, json_path, classes)

    # Save classes config for reference
    with open(output_dir / "classes.yaml", "w") as f:
        yaml.dump(classes, f)


if __name__ == "__main__":
    main()

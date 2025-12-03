"""Download a subset of Open Images (V7) for selected animals.

This script downloads images and annotations for specified animal classes
from the Open Images dataset. It uses a YAML configuration file to specify
the target classes, limit per class, and output directories.
"""

import argparse
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import requests
import yaml
from botocore.client import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

# Directory to save downloaded images
ROOT_DIR = Path(__file__).resolve().parents[2]
METADATA_DIR = (ROOT_DIR / "data" / "metadata").resolve()

# Open Images metadata URLs
BASE_URL = "https://storage.googleapis.com/openimages"
CLASS_URL = f"{BASE_URL}/v7/oidv7-class-descriptions-boxable.csv"

# Annotations URLs for train, validation, and test sets
ANNOTATIONS_URLS = {
    "train": f"{BASE_URL}/2018_04/train/train-annotations-bbox.csv",
    "validation": f"{BASE_URL}/2018_04/validation/validation-annotations-bbox.csv",
    "test": f"{BASE_URL}/2018_04/test/test-annotations-bbox.csv",
}


# Load the YAML config file
def load_config(config_path: str) -> dict[Any, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        result: dict[Any, Any] = yaml.safe_load(f)
        return result


# Function to download the dataset
def download_dataset(url: str, dest_path: Path) -> None:
    """Download a file from a URL to the specified destination path."""
    if dest_path.exists():
        print(f"File {dest_path} already exists. Skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(dest_path, "wb") as file,
        tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


# Function to get the list of image IDs for target classes
def get_image_ids_for_classes(target_classes: list, csv_path: Path) -> dict:
    """Get image IDs for target classes from the class descriptions CSV."""
    class_to_id = {}
    print("Searching for target IDs")

    df = pd.read_csv(csv_path, header=None, names=["LabelName", "DisplayName"])

    for target in target_classes:
        match = df[df["DisplayName"] == target]
        if not match.empty:
            class_id = match.iloc[0]["LabelName"]
            class_to_id[target] = class_id
            print(f"Found target class '{target}' with ID '{class_id}'")
        else:
            print(f"Warning: Target class '{target}' not found in class descriptions.")
    return class_to_id


# Function to filter annotations for target classes
def filter_annotations(
    class_to_id: dict, annotations_csv: Path, limit: int, split_name: str
) -> list:
    """Filter annotations to get image IDs up to the per-class limit."""
    print(f"Filtering {split_name} annotations for target classes")

    # Create reverse mapping: ID -> class name
    id_to_class = {v: k for k, v in class_to_id.items()}
    target_ids = set(class_to_id.values())

    relevant_images: list[str] = []
    class_counts = {class_name: 0 for class_name in class_to_id}

    chunk_size = 50000  # Reduced chunk size for better memory management
    chunk_num = 0

    try:
        # Use more efficient CSV reading with low_memory=False to avoid dtype warnings
        for chunk in pd.read_csv(
            annotations_csv,
            chunksize=chunk_size,
            usecols=["ImageID", "LabelName"],  # Only read necessary columns
            dtype={"ImageID": str, "LabelName": str},
        ):
            chunk_num += 1
            if chunk_num % 100 == 0:
                print(
                    "  Processed "
                    f"{chunk_num * chunk_size:,} rows... "
                    f"Found {len(relevant_images)} images so far"
                )

            # Filter rows with target labels
            filtered = chunk[chunk["LabelName"].isin(target_ids)]

            # Process filtered rows
            for label_id, group in filtered.groupby("LabelName"):
                class_name = id_to_class[label_id]

                # Get unique image IDs for this label
                image_ids = group["ImageID"].unique()

                for img_id in image_ids:
                    if class_counts[class_name] < limit:
                        relevant_images.append(img_id)
                        class_counts[class_name] += 1
                    else:
                        break

            # Check if we've reached the limit for all classes
            if all(count >= limit for count in class_counts.values()):
                print(f"Reached limit for all target classes in {split_name}.")
                break

    except Exception as e:
        print(f"Error processing annotations: {e}")
        raise

    unique_image_ids = list(set(relevant_images))
    print(f"Total unique images found for {split_name}: {len(unique_image_ids)}")
    print(f"Class distribution: {class_counts}")
    return unique_image_ids


# Function to download images based on image IDs
def download_images(image_ids: list, images_dir: Path, split_name: str) -> None:
    """Download images given a list of image IDs."""
    images_dir.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "open-images-dataset"

    print(f"Downloading {len(image_ids)} images for {split_name}...")
    success_count = 0

    for image_id in tqdm(image_ids, desc=f"{split_name} images"):
        # Images are stored in split-specific folders in S3
        image_key = f"{split_name}/{image_id}.jpg"
        local_path = images_dir / f"{image_id}.jpg"

        if local_path.exists():
            success_count += 1
            continue

        try:
            s3_client.download_file(bucket_name, image_key, str(local_path))
            success_count += 1
        except Exception as e:
            print(f"Error downloading image {image_id}: {e}")

    print(
        "Successfully downloaded "
        f"{success_count} out of {len(image_ids)} images "
        f"for {split_name}."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Download a subset of Open Images dataset "
        "for selected animal classes."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    """Run the dataset download process based on CLI arguments."""
    cfg = load_config(args.config)
    target_classes = cfg["dataset"]["classes"]
    limit = cfg["dataset"]["limit"]
    raw_dir_str = cfg["paths"]["raw_dir"]
    raw_dir = (ROOT_DIR / raw_dir_str).resolve()
    print(f"Start download with config: {args.config}")
    print(f" - Classes: {target_classes}")
    print(f" - Limit: {limit}")
    print(f" - Output Dir: {raw_dir}")
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    class_csv_path = METADATA_DIR / "oidv7-class-descriptions-boxable.csv"
    download_dataset(CLASS_URL, class_csv_path)
    class_to_id = get_image_ids_for_classes(target_classes, class_csv_path)
    if not class_to_id:
        print("Geen geldige klassen gevonden. Stoppen.")
        return 0

    # Track unique images across all splits so total reflects distinct files
    unique_image_ids: set[str] = set()
    for split_name, annotations_url in ANNOTATIONS_URLS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'=' * 60}")
        annotations_csv_path = METADATA_DIR / f"{split_name}-annotations-bbox.csv"
        download_dataset(annotations_url, annotations_csv_path)
        image_ids = filter_annotations(
            class_to_id, annotations_csv_path, limit, split_name
        )
        # Add new unique IDs only
        new_ids = [i for i in image_ids if i not in unique_image_ids]
        images_dir = raw_dir / "images" / split_name
        download_images(new_ids, images_dir, split_name)
        unique_image_ids.update(new_ids)
    print(f"\n{'=' * 60}")
    print("Dataset download complete!")
    print(f"{'=' * 60}")
    return len(unique_image_ids)


def main() -> None:  # pragma: no cover
    """Entry point for the raw dataset download CLI."""
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
ROOT_DIR = Path(__file__).resolve().parents[2]
dotenv_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path, override=False)

import fiftyone as fo
import fiftyone.zoo as foz
from minio import Minio


# Bucket name and prefix for datasets
BUCKET_NAME = "datasets"
BUCKET_PREFIX = "openimages_animals/raw/v1"

# Local dataset path
EXPORT_DIR = (ROOT_DIR / "data" / "openimages_animals").resolve()


# Function to create MinIO client
def get_minio_client() -> Minio:
    """Create and return a MinIO client using environment variables."""
    minio_endpoint = os.getenv("MINIO_ENDPOINT")
    minio_root_user = os.getenv("MINIO_ROOT_USER")
    minio_root_password = os.getenv("MINIO_ROOT_PASSWORD")

    is_missing = [
        name
        for name, value in [
            ("MINIO_ENDPOINT", minio_endpoint),
            ("MINIO_ROOT_USER", minio_root_user),
            ("MINIO_ROOT_PASSWORD", minio_root_password),
        ]
        if not value
    ]

    if is_missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(is_missing)}"
        )

    return Minio(
        endpoint=minio_endpoint,
        access_key=minio_root_user,
        secret_key=minio_root_password,
        secure=False,
    )


# Function to download the dataset
def download_dataset(max_samples: int = 300) -> fo.Dataset:
    """Download the Open Images dataset with specified parameters."""
    print("Downloading dataset from FiftyOne")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        label_types=["classifications", "detections"],
        classes=["Animal"],
        max_samples=max_samples,
    )
    return dataset


# Function to export dataset to local directory
def export_dataset_local(dataset: fo.Dataset, export_dir: Path = EXPORT_DIR) -> Path:
    """Export the dataset to the specified local directory."""
    export_dir.mkdir(parents=True, exist_ok=True)
    dataset.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.FiftyOneDataset,
        overwrite=True,
    )
    print(f"Dataset exported to {export_dir}")
    return export_dir


# Function to ensure bucket exists
def bucket_exists(minio_client: Minio, bucket_name: str) -> None:
    """Ensure the specified bucket exists in MinIO."""
    if not minio_client.bucket_exists(bucket_name=bucket_name):
        print(f"Creating bucket: {bucket_name}")
        minio_client.make_bucket(bucket_name=bucket_name)
    else:
        print(f"Bucket {bucket_name} already exists")


# Function to upload dataset to MinIO
def upload_dataset_minio(
    client: Minio,
    bucket_name: str,
    bucket_prefix: str,
    export_dir: Path = EXPORT_DIR,
) -> int:
    """Upload the dataset files to MinIO.
    Returns the number of files uploaded.
    """
    print(
        f"Uploading files from {export_dir} to bucket {bucket_name} with prefix {bucket_prefix}"
    )
    count = 0
    for file_path in export_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(export_dir).as_posix()
            object_name = f"{bucket_prefix}/{relative_path}"
            client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(file_path),
            )
            count += 1
            if count % 50 == 0:
                print(f"Uploaded {count} files...")

    print(f"Finished uploading {count} files to bucket {bucket_name}")
    return count


def main() -> None:
    """Main function to download, export, and upload dataset."""
    # Create MinIO client
    minio_client = get_minio_client()

    # Ensure bucket exists
    bucket_exists(minio_client, BUCKET_NAME)

    # Download dataset
    dataset = download_dataset(max_samples=300)

    # Export dataset locally
    export_dir = export_dataset_local(dataset, EXPORT_DIR)

    # Upload dataset to MinIO
    upload_dataset_minio(
        client=minio_client,
        bucket_name=BUCKET_NAME,
        bucket_prefix=BUCKET_PREFIX,
        export_dir=export_dir,
    )


if __name__ == "__main__":
    main()

"""Dataset download utilities for MinIO.

Handles downloading datasets from MinIO, exporting them locally.
"""

from pathlib import Path

from minio import Minio
from minio.error import S3Error

from src.data.minio_datamanagement import bucket_exists, get_minio_client

# Config keys for MinIO connection
BUCKET_NAME = "datasets"
BUCKET_PREFIX = "openimages_animals/raw/v1"  # Change if dataset version changes

# Local dataset path
ROOT_DIR = Path(__file__).resolve().parents[2]
LOCAL_DOWNLOAD_DIR = (ROOT_DIR / "data" / "openimages_animals").resolve()


# Function to download dataset from MinIO
def download_dataset_from_minio(
    client: Minio,
    bucket_name: str = BUCKET_NAME,
    bucket_prefix: str = BUCKET_PREFIX,
    local_download_dir: Path = LOCAL_DOWNLOAD_DIR,
) -> int:
    """Download dataset files from MinIO to local directory.

    Returns the number of files downloaded.
    """
    if not client.bucket_exists(bucket_name=bucket_name):
        raise S3Error(f"Bucket {bucket_name} does not exist in MinIO.")

    print(
        f"Downloading dataset from MinIO bucket: {bucket_name}, prefix: {bucket_prefix}"
    )

    # Ensure local download directory exists
    local_download_dir.mkdir(parents=True, exist_ok=True)

    # List all objects under the specified prefix
    objects = client.list_objects(
        bucket_name=bucket_name, prefix=bucket_prefix, recursive=True
    )

    download_count = 0
    for obj in objects:
        if obj.is_dir:
            continue  # Skip directories

        object_name = obj.object_name

        if object_name.startswith(bucket_prefix):
            relative_path = object_name[len(bucket_prefix) :].lstrip("/")
        else:
            relative_path = object_name

        local_file_path = local_download_dir / relative_path
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            client.fget_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(local_file_path),
            )
            download_count += 1
            print(f"Downloaded {object_name} to {local_file_path}")
        except S3Error as e:
            print(f"Error downloading {object_name}: {e}")

    print(f"Total files downloaded: {download_count}, to {local_download_dir}")
    return download_count


def main() -> None:
    """Main function to download dataset from MinIO."""
    try:
        minio_client = get_minio_client()
        bucket_exists(minio_client, BUCKET_NAME)
        download_dataset_from_minio(
            client=minio_client,
            bucket_name=BUCKET_NAME,
            bucket_prefix=BUCKET_PREFIX,
            local_download_dir=LOCAL_DOWNLOAD_DIR,
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

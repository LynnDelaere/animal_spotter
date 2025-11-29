"""Dataset upload utilities for MinIO.

This script uploads the locally downloaded OpenImages dataset (raw files)
to MinIO storage. It assumes the data has already been downloaded using
src/data/download_dataset_raw.py.
"""

from pathlib import Path

from minio import Minio

from src.data.minio_datamanagement import bucket_exists, get_minio_client

# Bucket name and prefix for datasets
BUCKET_NAME = "datasets"
BUCKET_PREFIX = "openimages_animals/raw/v1"

# Local dataset path
ROOT_DIR = Path(__file__).resolve().parents[2]
EXPORT_DIR = (ROOT_DIR / "data" / "openimages_animals").resolve()


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
        f"Uploading files from {export_dir} to bucket {bucket_name} "
        f"with prefix {bucket_prefix}"
    )
    count = 0
    for file_path in export_dir.rglob("*"):
        if file_path.is_file():
            if file_path.name.startswith("."):
                continue  # Skip hidden files
            relative_path = file_path.relative_to(export_dir).as_posix()
            object_name = f"{bucket_prefix}/{relative_path}"
            try:
                client.fput_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=str(file_path),
                )
                count += 1
                if count % 100 == 0:
                    print(f"Uploaded {count} files...")

            except Exception as e:
                print(f"Error uploading {file_path} to {object_name}: {e}")

    print(f"Finished uploading {count} files to bucket {bucket_name}")
    return count


def main() -> None:
    """Main function to validate dataset presence and upload it to MinIO."""
    if not EXPORT_DIR.exists() or not any(EXPORT_DIR.iterdir()):
        raise FileNotFoundError(
            f"Export directory {EXPORT_DIR} does not exist or is empty. "
            "Please download the dataset first using "
            "`src/data/download_dataset_raw.py`."
        )
    try:
        # Create MinIO client
        minio_client = get_minio_client()

        # Ensure bucket exists
        bucket_exists(minio_client, BUCKET_NAME)

        # Upload dataset to MinIO
        upload_dataset_minio(
            client=minio_client,
            bucket_name=BUCKET_NAME,
            bucket_prefix=BUCKET_PREFIX,
            export_dir=EXPORT_DIR,
        )
    except Exception as e:
        print(f"Error during dataset upload: {e}")


if __name__ == "__main__":
    main()

"""MinIO data management utilities.

Provides functions for creating MinIO clients and managing buckets.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio

# Load environment variables from .env file
ROOT_DIR = Path(__file__).resolve().parents[2]
dotenv_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path, override=False)


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
        raise OSError(
            f"Missing required environment variables: {', '.join(is_missing)}"
        )

    # Narrow Optional[str] to str for type checker
    assert minio_endpoint is not None
    assert minio_root_user is not None
    assert minio_root_password is not None

    return Minio(
        endpoint=minio_endpoint,
        access_key=minio_root_user,
        secret_key=minio_root_password,
        secure=False,
    )


# Function to ensure bucket exists
def bucket_exists(minio_client: Minio, bucket_name: str) -> None:
    """Ensure the specified bucket exists in MinIO."""
    if not minio_client.bucket_exists(bucket_name=bucket_name):
        print(f"Creating bucket: {bucket_name}")
        minio_client.make_bucket(bucket_name=bucket_name)
    else:
        print(f"Bucket {bucket_name} already exists")
    return

"""Download raw dataset from MinIO.

Handles downloading dataset files from MinIO bucket to local storage.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from minio import Minio

from src.data.minio_datamanagement import get_minio_client

# Local dataset path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


def load_config(path: str) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with cfg_path.open("r") as f:
        data: dict[str, Any] = yaml.safe_load(f)
        return data


def download_dataset_from_minio(
    client: Minio,
    bucket_name: str,
    bucket_prefix: str,
    local_download_dir: Path,
) -> int:
    """Download dataset files from MinIO bucket to local directory.

    Args:
        client: MinIO client instance
        bucket_name: Name of the MinIO bucket
        bucket_prefix: Prefix path in the bucket (e.g., 'openimages_animals/raw/v1')
        local_download_dir: Local directory to download files to

    Returns:
        Number of files downloaded

    Raises:
        ValueError: If bucket does not exist
    """
    if not client.bucket_exists(bucket_name=bucket_name):
        raise ValueError(f"Bucket '{bucket_name}' does not exist")

    print(
        f"Downloading from MinIO bucket '{bucket_name}' with prefix '{bucket_prefix}'"
    )
    print(f"Target directory: {local_download_dir}")

    # List all objects with the given prefix
    objects = client.list_objects(
        bucket_name=bucket_name, prefix=bucket_prefix, recursive=True
    )

    downloaded_count = 0
    for obj in objects:
        # Skip directory markers
        if obj.is_dir:
            continue

        object_name = obj.object_name

        # Calculate relative path by removing the prefix
        if object_name is not None and object_name.startswith(bucket_prefix):
            relative_path = object_name[len(bucket_prefix) :].lstrip("/")
        else:
            relative_path = object_name if object_name is not None else ""

        local_file_path = local_download_dir / relative_path

        # Create parent directories if needed
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        print(f"  Downloading: {object_name} -> {local_file_path}")
        if object_name is not None:
            client.fget_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(local_file_path),
            )
        downloaded_count += 1

    print(f"Downloaded {downloaded_count} files from MinIO")
    return downloaded_count


class _ProcessParser(argparse.ArgumentParser):
    def parse_args(  # type: ignore[override]
        self, args: list[str] | None = None, namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        ns = super().parse_args(args, namespace)
        if not ns.config and (not ns.bucket or not ns.prefix or not ns.local_dir):
            self.error(
                "--bucket, --prefix, and --local-dir are required when --config "
                "is not provided"
            )
        return ns


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = _ProcessParser(
        description="Download OpenImages Animals dataset from MinIO."
    )
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--bucket", type=str, help="MinIO bucket name")
    parser.add_argument("--prefix", type=str, help="Bucket prefix path")
    parser.add_argument("--local-dir", type=str, help="Local download directory")
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    """Run download and processing based on parsed arguments.

    Returns:
        Number of files downloaded from MinIO
    """
    if args.config:
        # Config mode
        cfg = load_config(args.config)
        ds_cfg = cfg.get("dataset", {})
        paths_cfg = cfg.get("paths", {})
        minio_cfg = cfg.get("minio", {})

        # Extract required config values
        raw_dir_str = paths_cfg.get("raw_dir")
        if not raw_dir_str:
            raise ValueError("Config must contain 'paths.raw_dir'")

        raw_prefix = minio_cfg.get("raw_prefix")
        if not raw_prefix:
            raise ValueError("Config must contain 'minio.raw_prefix'")

        bucket_name = minio_cfg.get("bucket", "datasets")
        version = ds_cfg.get("version", "v1")

        # Construct paths
        raw_dir_path = (ROOT_DIR / raw_dir_str).resolve()
        bucket_prefix_full = f"{raw_prefix}/{version}"

    else:
        # Explicit mode
        bucket_name = args.bucket
        bucket_prefix_full = args.prefix
        raw_dir_path = Path(args.local_dir).resolve()

    # Download from MinIO
    client = get_minio_client()
    downloaded_count = download_dataset_from_minio(
        client=client,
        bucket_name=bucket_name,
        bucket_prefix=bucket_prefix_full,
        local_download_dir=raw_dir_path,
    )

    return downloaded_count


def main() -> None:
    """Entry point for the download dataset CLI."""
    parser = build_parser()
    args = parser.parse_args()
    count = run_from_args(args)
    print(f"\nCompleted! Downloaded {count} files from MinIO.")


if __name__ == "__main__":
    main()

"""Generic script to upload datasets to MinIO storage.

Uploads a local directory (recursively) to a MinIO bucket using a YAML config
or CLI arguments.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml
from minio import Minio

from src.data.minio_datamanagement import bucket_exists, get_minio_client

# Local dataset path (default root)
ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(config_path: str) -> dict[Any, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        result: dict[Any, Any] = yaml.safe_load(f)
        return result


def upload_dataset_to_minio(
    client: Minio, local_path: Path, bucket_name: str, bucket_prefix: str
) -> int:
    """Recursively upload a directory to MinIO."""
    if not local_path.exists():
        raise FileNotFoundError(f"Local path {local_path} does not exist.")

    print(
        f"Uploading dataset from '{local_path}' to bucket '{bucket_name}'"
        f", prefix: '{bucket_prefix}'"
    )

    count = 0
    # rglob("*") pakt alles: images/, metadata/, etc.
    for file_path in local_path.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("."):
            # Relatief pad berekenen (bv. metadata/train.csv)
            rel_path = file_path.relative_to(local_path).as_posix()
            object_name = f"{bucket_prefix}/{rel_path}"

            try:
                client.fput_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=str(file_path),
                )
                count += 1
                if count % 100 == 0:
                    print(f"  Uploaded {count} files...")
            except Exception as e:
                print(f"Failed to upload {object_name}: {e}")

    print(f"Finished! Total uploaded files: {count}")
    return count


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    class CustomArgumentParser(argparse.ArgumentParser):
        def parse_args(  # type: ignore[override]
            self,
            args: list[str] | None = None,
            namespace: argparse.Namespace | None = None,
        ) -> argparse.Namespace:
            parsed = super().parse_args(args, namespace)
            # Validate that --prefix is provided when --config is not
            if not parsed.config and not parsed.prefix:
                self.error("--prefix is required when not using --config")
            return parsed

    parser = CustomArgumentParser(description="Upload dataset to MinIO.")
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    parser.add_argument("--local-path", help="Explicit local path (overrides config).")
    parser.add_argument("--prefix", help="Explicit bucket prefix (overrides config).")
    parser.add_argument(
        "--bucket",
        default="datasets",
        help="Bucket name (default: datasets).",
    )
    parser.add_argument(
        "--version", type=str, help="Dataset version (used if no config provided)."
    )
    parser.add_argument(
        "--type",
        choices=["raw", "processed", "both"],
        default="raw",
        help="Upload raw data, processed data, or both (default: raw).",
    )
    return parser


def _get_config_values(args: argparse.Namespace) -> tuple[str, dict | None, str | None]:
    """Extract config values and bucket name from arguments.

    Returns:
        Tuple of (bucket_name, config_dict, config_version)
    """
    bucket_name = args.bucket
    cfg = None
    cfg_version = None

    if args.config:
        cfg = load_config(args.config)
        cfg_version = (
            cfg.get("dataset", {}).get("version") if isinstance(cfg, dict) else None
        )
        if "minio" in cfg:
            bucket_name = cfg["minio"].get("bucket", bucket_name)

    return bucket_name, cfg, cfg_version


def _determine_paths(
    upload_type: str,
    cfg: dict | None,
    args: argparse.Namespace,
    effective_version: str | None,
) -> tuple[Path | None, str | None]:
    """Determine local path and bucket prefix for upload.

    Returns:
        Tuple of (local_path, bucket_prefix)
    """
    local_path = None
    bucket_prefix = None

    # Get paths from config
    if cfg:
        if "minio" in cfg:
            prefix_key = f"{upload_type}_prefix"
            bucket_prefix = cfg["minio"].get(prefix_key)

        if "paths" in cfg:
            path_key = f"{upload_type}_dir"
            dir_path = cfg["paths"].get(path_key)
            if dir_path:
                local_path = (ROOT_DIR / dir_path).resolve()
                # Add version to path if needed
                if effective_version and upload_type == "processed":
                    local_path = local_path / effective_version

    # CLI overrides
    if args.local_path:
        local_path_arg = Path(args.local_path)
        if local_path_arg.is_absolute():
            local_path = local_path_arg
        else:
            local_path = (ROOT_DIR / args.local_path).resolve()

    if args.prefix:
        bucket_prefix = args.prefix
    elif bucket_prefix and effective_version:
        bucket_prefix = f"{bucket_prefix}/{effective_version}"

    return local_path, bucket_prefix


def _validate_upload_paths(
    upload_type: str, local_path: Path | None, bucket_prefix: str | None
) -> bool:
    """Validate that paths are set and exist.

    Returns:
        True if validation passes, False otherwise
    """
    if not local_path or not bucket_prefix:
        print(f"ERROR: Could not determine 'local_path' or 'prefix' for {upload_type}.")
        print("Provide them via --config or CLI args.")
        return False

    print(f"DEBUG: Checking path {local_path}, exists={local_path.exists()}")
    if not local_path.exists():
        print(f"WARNING: Path {local_path} does not exist. Skipping {upload_type}.")
        return False

    return True


def _perform_sanity_checks(upload_type: str, local_path: Path) -> None:
    """Perform sanity checks based on upload type."""
    if upload_type == "raw":
        metadata_check = local_path / "metadata"
        if not metadata_check.exists():
            print(f"WARNING: No 'metadata' folder found in {local_path}!")
        else:
            print(f"Verified: 'metadata' folder found in {local_path}.")
    elif upload_type == "processed":
        json_files = list(local_path.glob("*.json"))
        if not json_files:
            print(f"WARNING: No JSON files found in {local_path}!")
        else:
            print(f"Verified: Found {len(json_files)} JSON file(s) in {local_path}.")


def run_from_args(args: argparse.Namespace) -> int:
    """Run the upload process based on CLI arguments."""
    # Get config values
    bucket_name, cfg, cfg_version = _get_config_values(args)

    # Validate required args when not using config
    if not args.config and not args.prefix:
        import sys

        print("Error: --prefix is required when not using --config", file=sys.stderr)
        sys.exit(2)

    effective_version = args.version or cfg_version
    minio_client = get_minio_client()
    bucket_exists(minio_client, bucket_name)

    total_uploaded = 0
    upload_types = ["raw", "processed"] if args.type == "both" else [args.type]

    for upload_type in upload_types:
        # Determine paths
        local_path, bucket_prefix = _determine_paths(
            upload_type, cfg, args, effective_version
        )

        # Validate paths
        if not _validate_upload_paths(upload_type, local_path, bucket_prefix):
            continue

        # After validation, these are guaranteed to be non-None
        assert local_path is not None
        assert bucket_prefix is not None

        # Sanity checks
        _perform_sanity_checks(upload_type, local_path)

        # Upload
        count = upload_dataset_to_minio(
            client=minio_client,
            local_path=local_path,
            bucket_name=bucket_name,
            bucket_prefix=bucket_prefix,
        )
        total_uploaded += count

    return total_uploaded


def main() -> None:
    """Entry point for the upload dataset CLI."""
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()

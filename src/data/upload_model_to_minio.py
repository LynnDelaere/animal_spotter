"""Generic script to upload trained models to MinIO storage.

Uploads a model directory (recursively) to a MinIO bucket using a YAML config
or CLI arguments.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml
from minio import Minio

from src.data.minio_datamanagement import bucket_exists, get_minio_client

# Local model path (default root)
ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(config_path: str) -> dict[Any, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        result: dict[Any, Any] = yaml.safe_load(f)
        return result


def upload_model_to_minio(
    client: Minio, local_path: Path, bucket_name: str, bucket_prefix: str
) -> int:
    """Recursively upload a model directory to MinIO."""
    if not local_path.exists():
        raise FileNotFoundError(f"Local path {local_path} does not exist.")

    print(
        f"Uploading model from '{local_path}' to bucket '{bucket_name}', "
        f"prefix: '{bucket_prefix}'"
    )

    count = 0
    # rglob("*") picks up everything: config.json, model.safetensors, etc.
    for file_path in local_path.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("."):
            # Calculate relative path (e.g., config.json, pytorch_model.bin)
            rel_path = file_path.relative_to(local_path).as_posix()
            object_name = f"{bucket_prefix}/{rel_path}"

            try:
                client.fput_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=str(file_path),
                )
                count += 1
                if count % 10 == 0:
                    print(f"  Uploaded {count} files...")
            except Exception as e:
                print(f"Failed to upload {object_name}: {e}")

    print(f"Finished! Total uploaded files: {count}")
    return count


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Upload trained model to MinIO.")
    parser.add_argument(
        "--config",
        type=str,
        default="dataset_v2.yaml",
        help="Path to YAML config file (default: dataset_v2.yaml).",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Model version (e.g., v1, v2). Overrides config version. Model will be "
        "loaded from models/detr-finetuned/<version>",
    )
    parser.add_argument(
        "--local-path",
        help="Explicit local path to model directory (overrides version-based lookup).",
    )
    parser.add_argument(
        "--bucket",
        default="models",
        help="Bucket name (default: models).",
    )
    parser.add_argument(
        "--prefix",
        help="Explicit bucket prefix (e.g., detr-finetuned/v1). If not provided, will "
        "be constructed from config.",
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
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT_DIR / "configs" / config_path

        if config_path.exists():
            cfg = load_config(str(config_path))
            cfg_version = (
                cfg.get("dataset", {}).get("version") if isinstance(cfg, dict) else None
            )

    return bucket_name, cfg, cfg_version


def _determine_model_path(
    version: str | None,
    local_path: str | None,
) -> Path | None:
    """Determine the local model path.

    Returns:
        Path to the model directory or None if not found
    """
    local_model_path = None

    # If explicit local path is provided, use it
    if local_path:
        local_path_obj = Path(local_path)
        if local_path_obj.is_absolute():
            local_model_path = local_path_obj
        else:
            local_model_path = (ROOT_DIR / local_path).resolve()
    elif version:
        # Otherwise, construct path from version
        local_model_path = ROOT_DIR / "models" / "detr-finetuned" / version

    return local_model_path


def _determine_prefix(
    version: str | None,
    prefix: str | None,
) -> str | None:
    """Determine the bucket prefix for upload.

    Returns:
        Bucket prefix (e.g., 'detr-finetuned/v2') or None
    """
    if prefix:
        return prefix

    if version:
        return f"detr-finetuned/{version}"

    return None


def _validate_upload_path(model_path: Path | None) -> bool:
    """Validate that model path is set and exists.

    Returns:
        True if validation passes, False otherwise
    """
    if not model_path:
        print("ERROR: Could not determine model path.")
        print("Provide it via --version, --local-path, or --config.")
        return False

    if not model_path.exists():
        print(f"ERROR: Model path {model_path} does not exist.")
        return False

    return True


def _perform_sanity_checks(model_path: Path) -> None:
    """Perform sanity checks on model directory."""
    required_files = ["config.json", "model.safetensors"]
    found_files = []

    for required_file in required_files:
        if (model_path / required_file).exists():
            found_files.append(required_file)

    if found_files:
        print(
            f"Verified: Found {len(found_files)} expected file(s) in model directory:"
        )
        for f in found_files:
            print(f"  - {f}")
    else:
        print(
            f"WARNING: No expected model files found in {model_path}. "
            "Expected: config.json, model.safetensors"
        )


def run_from_args(args: argparse.Namespace) -> int:
    """Run the upload process based on CLI arguments."""
    # Get config values
    bucket_name, cfg, cfg_version = _get_config_values(args)

    # Determine effective version (from CLI arg or config)
    effective_version = args.version or cfg_version

    # Validate that we have a version
    if not effective_version and not args.local_path:
        print("ERROR: Could not determine model version.")
        print("Provide it via --version, --config, or --local-path.")
        return 0

    # Determine model path
    model_path = _determine_model_path(effective_version, args.local_path)

    # Validate model path
    if not _validate_upload_path(model_path):
        return 0

    # Model path is guaranteed to be non-None at this point
    assert model_path is not None

    # Determine bucket prefix
    bucket_prefix = _determine_prefix(effective_version, args.prefix)

    if not bucket_prefix:
        print("ERROR: Could not determine bucket prefix.")
        print("Provide it via --prefix or --version/--config.")
        return 0

    # Perform sanity checks
    _perform_sanity_checks(model_path)

    # Ensure bucket exists
    minio_client = get_minio_client()
    bucket_exists(minio_client, bucket_name)

    # Upload
    count = upload_model_to_minio(
        client=minio_client,
        local_path=model_path,
        bucket_name=bucket_name,
        bucket_prefix=bucket_prefix,
    )

    return count


def main() -> None:
    """Entry point for the upload model CLI."""
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()

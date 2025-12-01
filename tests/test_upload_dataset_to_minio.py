"""Tests for dataset upload functionality and CLI parser."""

from pathlib import Path
from typing import Any, cast

import pytest
import src.data.upload_dataset_to_minio as mod
from minio import Minio
from src.data.upload_dataset_to_minio import (
    build_parser,
    run_from_args,
    upload_dataset_to_minio,
)

mod.get_minio_client = lambda: cast(Any, FakeMinio())  # type: ignore[assignment]
mod.bucket_exists = lambda _c, _b: True  # type: ignore[assignment]


# A simple fake Minio client for testing purposes
class FakeMinio:
    """Simple fake Minio client for testing purposes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.buckets: set[str] = {"datasets"}
        self.uploaded: list[tuple[str, str, str]] = []

    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if a bucket exists."""
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        """Create a new bucket."""
        self.buckets.add(bucket_name)

    def fput_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        """Upload a file to a bucket."""
        assert Path(file_path).is_file()
        self.uploaded.append((bucket_name, object_name, file_path))


def test_upload_export_dir_to_minio(tmp_path: Path) -> None:
    """Test uploading files from export_dir to MinIO."""
    # Create a fake directory structure with files
    export_dir = tmp_path / "exported_dataset"
    (export_dir / "subdir").mkdir(parents=True)

    img1 = export_dir / "image1.jpg"
    img1.write_bytes(b"fake image data 1")

    label1 = export_dir / "subdir" / "label1.txt"
    label1.write_text("fake label data 1")

    # Create a fake MinIO client
    fake_minio = FakeMinio()
    bucket_name = "test-bucket"
    prefix = "test-prefix"

    # Call the upload function
    uploaded_count = upload_dataset_to_minio(
        client=cast(Minio, fake_minio),
        local_path=export_dir,
        bucket_name=bucket_name,
        bucket_prefix=prefix,
    )

    # Verify the number of uploaded files
    assert uploaded_count == 2
    assert len(fake_minio.uploaded) == 2

    # Verify the opbject names and file paths
    expected_uploads = sorted(obj[1] for obj in fake_minio.uploaded)
    assert expected_uploads == [
        f"{prefix}/image1.jpg",
        f"{prefix}/subdir/label1.txt",
    ]

    # Check the bucket names
    for bname, _object_name, file_path in fake_minio.uploaded:
        assert bname == bucket_name
        assert Path(file_path).is_file()


def test_cli_upload_minio_args(tmp_path: Path) -> None:
    """Test CLI parser for upload_dataset_to_minio with explicit args."""
    # Create a fake dataset directory
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "file1.txt").write_text("data1")
    (dataset_dir / "file2.txt").write_text("data2")

    parser = build_parser()
    args = parser.parse_args(
        [
            "--local-path",
            str(dataset_dir),
            "--prefix",
            "test-prefix",
            "--bucket",
            "test-bucket",
        ]
    )

    uploaded_count = run_from_args(args)
    assert uploaded_count == 2


def test_cli_upload_requires_prefix(tmp_path: Path) -> None:
    """Test that CLI parser raises error when prefix is missing."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--local-path", str(dataset_dir)])


def test_cli_upload_config_mode(tmp_path: Path) -> None:
    """Test CLI upload using config file mode."""
    # Minimal config
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "dataset:\n  version: vX\n"
        "paths:\n  raw_dir: data/rawtest\n"
        "minio:\n  bucket: datasets\n"
        "  raw_prefix: openimages_animals/raw\n"
    )

    # Create local dir expected by config
    local_dir = Path(__file__).resolve().parents[1] / "data" / "rawtest"
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "img.jpg").write_text("x")

    parser = build_parser()
    args = parser.parse_args(["--config", str(cfg)])
    uploaded = run_from_args(args)

    # Only one file present
    assert uploaded == 1

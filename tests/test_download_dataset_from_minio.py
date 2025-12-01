"""Tests for download_dataset_from_minio functionality and CLI parser."""

from pathlib import Path
from typing import Any, cast

import pytest
import src.data.download_dataset_from_minio as mod
from minio import Minio
from src.data.download_dataset_from_minio import (
    build_parser,
    download_dataset_from_minio,
    run_from_args,
)


class FakeMinio:
    """Simple fake Minio client for testing purposes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.buckets: set[str] = {"datasets", "test-bucket"}
        self.objects: dict[str, list[tuple[str, bool]]] = {}

    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if a bucket exists."""
        return bucket_name in self.buckets

    def list_objects(self, bucket_name: str, prefix: str, recursive: bool) -> list[Any]:
        """List objects in a bucket with a given prefix."""

        class FakeObject:
            def __init__(self, name: str, is_dir: bool = False):
                self.object_name = name
                self.is_dir = is_dir

        # Return objects stored for this bucket/prefix combo
        key = f"{bucket_name}:{prefix}"
        if key in self.objects:
            return [FakeObject(name, is_dir) for name, is_dir in self.objects[key]]

        # Default behavior: simulate two objects
        return [
            FakeObject(f"{prefix}/a.jpg"),
            FakeObject(f"{prefix}/b.jpg"),
        ]

    def fget_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        """Download a file from a bucket."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_text("fake content")

    def add_objects(self, bucket_name: str, prefix: str, objects: list[str]) -> None:
        """Helper method to add objects for testing."""
        key = f"{bucket_name}:{prefix}"
        self.objects[key] = [(obj, False) for obj in objects]


# Patch points for CLI tests
mod.get_minio_client = lambda: cast(Any, FakeMinio())  # type: ignore[assignment]


def test_download_dataset_from_minio(tmp_path: Path) -> None:
    """Test downloading files from MinIO to local directory."""
    # Create fake MinIO client with custom objects
    fake_minio = FakeMinio()
    bucket_name = "test-bucket"
    prefix = "test-prefix"

    # Add test objects
    fake_minio.add_objects(
        bucket_name,
        prefix,
        [
            f"{prefix}/image1.jpg",
            f"{prefix}/subdir/image2.jpg",
            f"{prefix}/subdir/label.txt",
        ],
    )

    local_dir = tmp_path / "downloads"

    # Call the download function
    downloaded_count = download_dataset_from_minio(
        client=cast(Minio, fake_minio),
        bucket_name=bucket_name,
        bucket_prefix=prefix,
        local_download_dir=local_dir,
    )

    # Verify the number of downloaded files
    assert downloaded_count == 3

    # Verify files exist locally
    assert (local_dir / "image1.jpg").is_file()
    assert (local_dir / "subdir" / "image2.jpg").is_file()
    assert (local_dir / "subdir" / "label.txt").is_file()


def test_download_dataset_from_minio_empty_prefix(tmp_path: Path) -> None:
    """Test downloading when no files match the prefix."""
    fake_minio = FakeMinio()
    bucket_name = "test-bucket"
    prefix = "empty-prefix"

    # Add no objects for this prefix
    fake_minio.add_objects(bucket_name, prefix, [])

    local_dir = tmp_path / "downloads"

    downloaded_count = download_dataset_from_minio(
        client=cast(Minio, fake_minio),
        bucket_name=bucket_name,
        bucket_prefix=prefix,
        local_download_dir=local_dir,
    )

    assert downloaded_count == 0


def test_download_dataset_from_minio_nonexistent_bucket(tmp_path: Path) -> None:
    """Test that downloading from non-existent bucket raises error."""
    fake_minio = FakeMinio()
    bucket_name = "nonexistent-bucket"
    prefix = "test-prefix"
    local_dir = tmp_path / "downloads"

    with pytest.raises(ValueError, match="does not exist"):
        download_dataset_from_minio(
            client=cast(Minio, fake_minio),
            bucket_name=bucket_name,
            bucket_prefix=prefix,
            local_download_dir=local_dir,
        )


def test_download_dataset_from_minio_skips_directories(tmp_path: Path) -> None:
    """Test that directory objects are skipped during download."""

    class FakeMinioWithDirs(FakeMinio):
        def list_objects(
            self, bucket_name: str, prefix: str, recursive: bool
        ) -> list[Any]:
            class FakeObject:
                def __init__(self, name: str, is_dir: bool = False):
                    self.object_name = name
                    self.is_dir = is_dir

            return [
                FakeObject(f"{prefix}/dir/", is_dir=True),
                FakeObject(f"{prefix}/file.txt", is_dir=False),
            ]

    fake_minio = FakeMinioWithDirs()
    bucket_name = "test-bucket"
    prefix = "test-prefix"
    local_dir = tmp_path / "downloads"

    downloaded_count = download_dataset_from_minio(
        client=cast(Minio, fake_minio),
        bucket_name=bucket_name,
        bucket_prefix=prefix,
        local_download_dir=local_dir,
    )

    # Should only download the file, not the directory
    assert downloaded_count == 1
    assert (local_dir / "file.txt").is_file()


def test_cli_download_minio_explicit(tmp_path: Path) -> None:
    """Test CLI download with explicit arguments."""
    parser = build_parser()
    local_dir = tmp_path / "dl"
    args = parser.parse_args(
        [
            "--bucket",
            "datasets",
            "--prefix",
            "openimages_animals/raw/v1",
            "--local-dir",
            str(local_dir),
        ]
    )
    count = run_from_args(args)
    assert (local_dir / "a.jpg").is_file()
    assert (local_dir / "b.jpg").is_file()
    assert count == 2


def test_cli_download_minio_requires_arguments() -> None:
    """Test that CLI download requires prefix and local-dir arguments."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--bucket", "datasets"])  # missing prefix/local-dir


def test_cli_download_minio_config_mode(tmp_path: Path) -> None:
    """Test CLI download using config file mode."""
    # Create minimal config
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "dataset:\n  version: vX\n"
        "paths:\n  raw_dir: data/raw_download_test\n"
        "minio:\n  bucket: datasets\n"
        "  raw_prefix: openimages_animals/raw\n"
    )

    # Create expected local directory
    local_dir = Path(__file__).resolve().parents[2] / "data" / "raw_download_test"
    local_dir.mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    args = parser.parse_args(["--config", str(cfg)])
    count = run_from_args(args)

    # Should download the default 2 files (a.jpg, b.jpg)
    assert count == 2


def test_cli_download_minio_config_missing_prefix(tmp_path: Path) -> None:
    """Test that config mode requires raw_prefix."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "dataset:\n  version: vX\n"
        "paths:\n  raw_dir: data/raw_test\n"
        "minio:\n  bucket: datasets\n"
    )

    parser = build_parser()
    args = parser.parse_args(["--config", str(cfg)])

    with pytest.raises(ValueError, match="raw_prefix"):
        run_from_args(args)


def test_cli_download_minio_config_missing_raw_dir(tmp_path: Path) -> None:
    """Test that config mode requires raw_dir."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "dataset:\n  version: vX\n"
        "minio:\n  bucket: datasets\n"
        "  raw_prefix: openimages_animals/raw\n"
    )

    parser = build_parser()
    args = parser.parse_args(["--config", str(cfg)])

    with pytest.raises(ValueError, match="raw_dir"):
        run_from_args(args)


def test_cli_download_minio_requires_prefix_without_config(tmp_path: Path) -> None:
    """Test that explicit mode requires both prefix and local-dir."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        args = parser.parse_args(["--local-dir", str(tmp_path)])
        run_from_args(args)

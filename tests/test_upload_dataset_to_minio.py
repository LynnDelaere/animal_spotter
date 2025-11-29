"""Tests for dataset upload functionality."""

from pathlib import Path
from typing import Any, cast

from minio import Minio
from src.data.upload_dataset_to_minio import upload_dataset_minio


# A simple fake Minio client for testing purposes
class FakeMinio:
    """Simple fake Minio client for testing purposes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.buckets: set[str] = set()
        # keep track of uploaded objects like the real client would
        self.uploaded: list[tuple[str, str, str]] = []

    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if a bucket exists."""
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        """Create a new bucket."""
        self.buckets.add(bucket_name)

    def fput_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        """Upload a file to a bucket."""
        # Simulate uploading by just printing
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
    uploaded_count = upload_dataset_minio(
        client=cast(Minio, fake_minio),
        bucket_name=bucket_name,
        bucket_prefix=prefix,
        export_dir=export_dir,
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

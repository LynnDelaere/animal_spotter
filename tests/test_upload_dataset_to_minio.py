import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.upload_dataset_to_minio import get_minio_client, upload_dataset_minio


class FakeMinio:
    """Simple fake Minio client for testing purposes."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.buckets = set()
        # keep track of uploaded objects like the real client would
        self.uploaded = []

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        self.buckets.add(bucket_name)

    def fput_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        # Simulate uploading by just printing
        assert Path(file_path).is_file()
        self.uploaded.append((bucket_name, object_name, file_path))

def test_get_minio_client_raises_env_error(monkeypatch):
    """Test that get_minio_client raises an error if env vars are missing."""
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    # The implementation expects these variable names
    monkeypatch.delenv("MINIO_ROOT_USER", raising=False)
    monkeypatch.delenv("MINIO_ROOT_PASSWORD", raising=False)

    from src.data import upload_dataset_to_minio as upload_module
    # The current implementation raises ValueError when required env vars are missing
    with pytest.raises(EnvironmentError) as exinfo:
        upload_module.get_minio_client()

    msg = str(exinfo.value)
    assert "MINIO_ENDPOINT" in msg
    assert "MINIO_ROOT_USER" in msg
    assert "MINIO_ROOT_PASSWORD" in msg

def test_upload_export_dir_to_minio(tmp_path):
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
        client=fake_minio,
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
    for bname, object_name, file_path in fake_minio.uploaded:
        assert bname == bucket_name
        assert Path(file_path).is_file()

"""Tests for MinIO data management functionality."""

from typing import cast

import pytest
from minio import Minio
from src.data.upload_dataset_to_minio import bucket_exists, get_minio_client


# A mock Minio client for testing purposes
class MockMinioClient:
    def __init__(self) -> None:
        self.buckets = set[str]()
        self.created_buckets = list[str]()

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        self.buckets.add(bucket_name)
        self.created_buckets.append(bucket_name)


# Test function to verify get_minio_client behavior
def test_get_minio_client_raises_env_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_minio_client raises an error if env vars are missing."""
    # Remove environment variables to simulate missing configuration
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ROOT_USER", raising=False)
    monkeypatch.delenv("MINIO_ROOT_PASSWORD", raising=False)

    # The implementation should raise EnvironmentError
    with pytest.raises(EnvironmentError) as exinfo:
        get_minio_client()

    msg = str(exinfo.value)
    # Check that the error message contains the names of the missing variables
    assert "MINIO_ENDPOINT" in msg
    assert "MINIO_ROOT_USER" in msg
    assert "MINIO_ROOT_PASSWORD" in msg


# Test function to verify bucket_exists behavior
def test_bucket_exists_creates_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that bucket_exists creates a bucket if it doesn't exist."""
    mock_client = MockMinioClient()
    test_bucket_name = "test-bucket"

    # Ensure the bucket does not exist initially
    assert not mock_client.bucket_exists(test_bucket_name)

    # Call the function to ensure the bucket exists
    bucket_exists(cast(Minio, mock_client), test_bucket_name)

    # Now the bucket should exist
    assert mock_client.bucket_exists(test_bucket_name)
    # And it should have been created
    assert test_bucket_name in mock_client.created_buckets

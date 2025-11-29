import sys
import pytest
from pathlib import Path

# Ensure src is in sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the functions to be tested
from src.data.upload_dataset_to_minio import get_minio_client

# Test get_minio_client function
def test_get_minio_client_raises_env_error(monkeypatch):
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

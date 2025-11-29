"""Tests for download_dataset_raw.py functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from src.data.download_dataset_raw import (
    download_dataset,
    download_images,
    filter_annotations,
    get_image_ids_for_classes,
)


def test_download_dataset(tmp_path: Path) -> None:
    """Test download_dataset with a mocked requests.get."""
    url = "http://example.com/file.csv"
    dest_path = tmp_path / "file.csv"
    mock_response = MagicMock()
    mock_response.headers = {"content-length": "4"}
    mock_response.iter_content.return_value = [b"data"]
    with patch("requests.get", return_value=mock_response):
        download_dataset(url, dest_path)
    assert dest_path.exists()
    assert dest_path.read_bytes() == b"data"


def test_get_image_ids_for_classes(tmp_path: Path) -> None:
    """Test get_image_ids_for_classes with a small CSV."""
    csv_path = tmp_path / "classes.csv"
    df = pd.DataFrame(
        {
            "LabelName": ["/m/01", "/m/02"],
            "DisplayName": ["Cat", "Dog"],
        }
    )
    df.to_csv(csv_path, index=False, header=False)
    result = get_image_ids_for_classes(["Cat", "Dog", "Horse"], csv_path)
    assert result["Cat"] == "/m/01"
    assert result["Dog"] == "/m/02"
    assert "Horse" not in result


def test_filter_annotations(tmp_path: Path) -> None:
    """Test filter_annotations with a small CSV."""
    class_to_id = {"Cat": "/m/01", "Dog": "/m/02"}
    csv_path = tmp_path / "annots.csv"
    df = pd.DataFrame(
        {
            "ImageID": ["img1", "img2", "img3", "img4"],
            "LabelName": ["/m/01", "/m/02", "/m/01", "/m/02"],
        }
    )
    df.to_csv(csv_path, index=False)
    ids = filter_annotations(class_to_id, csv_path, limit=1, split_name="test")
    # Should only get one image per class
    assert len(ids) == 2
    assert set(ids) <= {"img1", "img2", "img3", "img4"}


def test_download_images(tmp_path: Path) -> None:
    """Test download_images with a mocked S3 client."""
    image_ids = ["img1", "img2"]
    images_dir = tmp_path / "images"
    split_name = "test"
    # Patch boto3 client and its download_file method
    with patch("boto3.client") as mock_client:
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        download_images(image_ids, images_dir, split_name)
        # Should call download_file for each image
        assert mock_s3.download_file.call_count == 2
        actual_calls = [tuple(call[0]) for call in mock_s3.download_file.call_args_list]
        assert actual_calls == [
            ("open-images-dataset", "test/img1.jpg", str(images_dir / "img1.jpg")),
            ("open-images-dataset", "test/img2.jpg", str(images_dir / "img2.jpg")),
        ]

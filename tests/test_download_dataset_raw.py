"""Tests for download_dataset_raw functionality and CLI parser."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import src.data.download_dataset_raw as mod
from src.data.download_dataset_raw import (
    build_parser,
    download_dataset,
    download_images,
    filter_annotations,
    get_image_ids_for_classes,
    run_from_args,
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


def test_cli_download_raw_config_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test CLI download using config file mode."""
    # Prepare minimal config
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "dataset:\n  classes: [Cat]\n  limit: 1\npaths:\n  raw_dir: data/raw_cli_test\n"
    )

    # Create expected local directory structure
    local_dir = Path(__file__).resolve().parents[2] / "data" / "raw_cli_test"
    (local_dir / "images" / "train").mkdir(parents=True, exist_ok=True)

    # Patch external functions to avoid network/S3
    monkeypatch.setattr(mod, "download_dataset", lambda *a, **k: None)
    monkeypatch.setattr(
        mod, "get_image_ids_for_classes", lambda *a, **k: {"Cat": "/m/01"}
    )
    monkeypatch.setattr(mod, "filter_annotations", lambda *a, **k: ["img1"])  # one id
    monkeypatch.setattr(mod, "download_images", lambda *a, **k: None)

    parser = build_parser()
    args = parser.parse_args(["--config", str(cfg)])
    total = run_from_args(args)
    assert total == 1


def test_cli_download_raw_requires_config() -> None:
    """Test that CLI download requires a config argument."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

"""Tests for convert_to_coco functionality."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from src.data.convert_to_coco import (
    create_coco_json,
    get_class_mapping,
    load_config,
)


@pytest.fixture
def mock_metadata_dir(tmp_path: Path) -> Path:
    """Create a mock metadata directory with class descriptions."""
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    # Create class descriptions CSV
    desc_csv = metadata_dir / "oidv7-class-descriptions-boxable.csv"
    df = pd.DataFrame(
        {
            "LabelName": ["/m/01yrx", "/m/0bt9lr", "/m/03k3r"],
            "DisplayName": ["Cat", "Dog", "Horse"],
        }
    )
    df.to_csv(desc_csv, index=False, header=False)

    return metadata_dir


@pytest.fixture
def mock_annotations_csv(tmp_path: Path) -> Path:
    """Create a mock annotations CSV."""
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    annot_csv = metadata_dir / "train-annotations-bbox.csv"
    df = pd.DataFrame(
        {
            "ImageID": ["img001", "img001", "img002"],
            "LabelName": ["/m/01yrx", "/m/0bt9lr", "/m/01yrx"],
            "XMin": [0.1, 0.2, 0.3],
            "XMax": [0.5, 0.6, 0.7],
            "YMin": [0.1, 0.2, 0.3],
            "YMax": [0.5, 0.6, 0.7],
        }
    )
    df.to_csv(annot_csv, index=False)

    return metadata_dir


@pytest.fixture
def mock_images_dir(tmp_path: Path) -> Path:
    """Create a mock images directory with dummy images."""
    images_dir = tmp_path / "images" / "train"
    images_dir.mkdir(parents=True)

    # Create dummy image files using PIL
    from PIL import Image

    for img_id in ["img001", "img002"]:
        img_path = images_dir / f"{img_id}.jpg"
        img = Image.new("RGB", (640, 480), color="red")
        img.save(img_path)

    return images_dir.parent


def test_load_config(tmp_path: Path) -> None:
    """Test load_config reads YAML correctly."""
    config_path = tmp_path / "config.yaml"
    config_data = {
        "dataset": {"version": "v1", "classes": ["Cat", "Dog"]},
        "paths": {"raw_dir": "data/raw"},
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    result = load_config(str(config_path))
    assert result["dataset"]["version"] == "v1"
    assert result["dataset"]["classes"] == ["Cat", "Dog"]


def test_load_config_file_not_found() -> None:
    """Test load_config raises error for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_get_class_mapping(mock_metadata_dir: Path) -> None:
    """Test get_class_mapping creates correct mappings."""
    target_classes = ["Cat", "Dog"]
    mid_to_name, name_to_id = get_class_mapping(mock_metadata_dir, target_classes)

    assert mid_to_name["/m/01yrx"] == "Cat"
    assert mid_to_name["/m/0bt9lr"] == "Dog"
    assert name_to_id["Cat"] == 0
    assert name_to_id["Dog"] == 1


def test_get_class_mapping_missing_classes(mock_metadata_dir: Path, capsys) -> None:
    """Test get_class_mapping warns about missing classes."""
    target_classes = ["Cat", "Dog", "Elephant"]
    mid_to_name, name_to_id = get_class_mapping(mock_metadata_dir, target_classes)

    captured = capsys.readouterr()
    assert "WARNING: Classes not found" in captured.out
    assert "Elephant" in captured.out
    assert "/m/01yrx" in mid_to_name
    assert "/m/0bt9lr" in mid_to_name


def test_get_class_mapping_missing_file(tmp_path: Path) -> None:
    """Test get_class_mapping raises error for missing metadata file."""
    with pytest.raises(FileNotFoundError):
        get_class_mapping(tmp_path, ["Cat"])


def test_create_coco_json(
    tmp_path: Path, mock_metadata_dir: Path, mock_images_dir: Path
) -> None:
    """Test create_coco_json generates valid COCO format."""
    # Setup annotations
    annot_csv = mock_metadata_dir / "train-annotations-bbox.csv"
    df = pd.DataFrame(
        {
            "ImageID": ["img001", "img001", "img002"],
            "LabelName": ["/m/01yrx", "/m/0bt9lr", "/m/01yrx"],
            "XMin": [0.1, 0.2, 0.3],
            "XMax": [0.5, 0.6, 0.7],
            "YMin": [0.1, 0.2, 0.3],
            "YMax": [0.5, 0.6, 0.7],
        }
    )
    df.to_csv(annot_csv, index=False)

    output_path = tmp_path / "output" / "train.json"
    target_classes = ["Cat", "Dog"]

    create_coco_json(
        split="train",
        metadata_dir=mock_metadata_dir,
        images_dir=mock_images_dir,
        output_path=output_path,
        target_classes=target_classes,
    )

    # Verify output exists
    assert output_path.exists()

    # Load and validate COCO format
    with open(output_path) as f:
        coco_data = json.load(f)

    # Check structure
    assert "info" in coco_data
    assert "images" in coco_data
    assert "annotations" in coco_data
    assert "categories" in coco_data

    # Check images
    assert len(coco_data["images"]) == 2
    assert coco_data["images"][0]["width"] == 640
    assert coco_data["images"][0]["height"] == 480

    # Check categories
    assert len(coco_data["categories"]) == 2
    assert coco_data["categories"][0]["name"] == "Cat"
    assert coco_data["categories"][1]["name"] == "Dog"

    # Check annotations
    assert len(coco_data["annotations"]) == 3

    # Verify bbox format [x, y, width, height]
    ann = coco_data["annotations"][0]
    assert "bbox" in ann
    assert len(ann["bbox"]) == 4
    assert ann["bbox"][0] == 0.1 * 640  # x_min
    assert ann["bbox"][1] == 0.1 * 480  # y_min
    assert ann["bbox"][2] == (0.5 - 0.1) * 640  # width
    assert ann["bbox"][3] == (0.5 - 0.1) * 480  # height

    # Verify annotation fields
    assert "id" in ann
    assert "image_id" in ann
    assert "category_id" in ann
    assert "area" in ann
    assert "iscrowd" in ann


def test_create_coco_json_no_images(
    tmp_path: Path, mock_metadata_dir: Path, capsys
) -> None:
    """Test create_coco_json handles missing images directory."""
    empty_images_dir = tmp_path / "empty_images"
    empty_images_dir.mkdir()

    output_path = tmp_path / "output" / "train.json"

    create_coco_json(
        split="train",
        metadata_dir=mock_metadata_dir,
        images_dir=empty_images_dir,
        output_path=output_path,
        target_classes=["Cat", "Dog"],
    )

    captured = capsys.readouterr()
    assert "Skipping train" in captured.out


def test_create_coco_json_filters_classes(
    tmp_path: Path, mock_metadata_dir: Path, mock_images_dir: Path
) -> None:
    """Test create_coco_json only includes specified classes."""
    # Setup annotations with multiple classes
    annot_csv = mock_metadata_dir / "train-annotations-bbox.csv"
    df = pd.DataFrame(
        {
            "ImageID": ["img001", "img001", "img002"],
            "LabelName": ["/m/01yrx", "/m/03k3r", "/m/01yrx"],  # Cat, Horse, Cat
            "XMin": [0.1, 0.2, 0.3],
            "XMax": [0.5, 0.6, 0.7],
            "YMin": [0.1, 0.2, 0.3],
            "YMax": [0.5, 0.6, 0.7],
        }
    )
    df.to_csv(annot_csv, index=False)

    output_path = tmp_path / "output" / "train.json"
    target_classes = ["Cat"]  # Only Cat, not Horse

    create_coco_json(
        split="train",
        metadata_dir=mock_metadata_dir,
        images_dir=mock_images_dir,
        output_path=output_path,
        target_classes=target_classes,
    )

    with open(output_path) as f:
        coco_data = json.load(f)

    # Should only have Cat annotations
    assert len(coco_data["annotations"]) == 2
    assert len(coco_data["categories"]) == 1
    assert coco_data["categories"][0]["name"] == "Cat"


def test_create_coco_json_handles_corrupt_image(
    tmp_path: Path, mock_metadata_dir: Path, capsys
) -> None:
    """Test create_coco_json handles corrupt/unreadable images."""
    images_dir = tmp_path / "images" / "train"
    images_dir.mkdir(parents=True)

    # Create a corrupt image file
    corrupt_img = images_dir / "corrupt.jpg"
    corrupt_img.write_text("not a valid image")

    # Create annotations
    annot_csv = mock_metadata_dir / "train-annotations-bbox.csv"
    df = pd.DataFrame(
        {
            "ImageID": ["corrupt"],
            "LabelName": ["/m/01yrx"],
            "XMin": [0.1],
            "XMax": [0.5],
            "YMin": [0.1],
            "YMax": [0.5],
        }
    )
    df.to_csv(annot_csv, index=False)

    output_path = tmp_path / "output" / "train.json"

    create_coco_json(
        split="train",
        metadata_dir=mock_metadata_dir,
        images_dir=images_dir.parent,
        output_path=output_path,
        target_classes=["Cat"],
    )

    captured = capsys.readouterr()
    assert "Error reading" in captured.out

    # Should still create output file but with no images/annotations
    with open(output_path) as f:
        coco_data = json.load(f)

    assert len(coco_data["images"]) == 0
    assert len(coco_data["annotations"]) == 0


def test_create_coco_json_category_ids_match_annotations(
    tmp_path: Path, mock_metadata_dir: Path, mock_images_dir: Path
) -> None:
    """Test that category IDs in annotations match the categories list."""
    # Setup annotations
    annot_csv = mock_metadata_dir / "train-annotations-bbox.csv"
    df = pd.DataFrame(
        {
            "ImageID": ["img001"],
            "LabelName": ["/m/0bt9lr"],  # Dog
            "XMin": [0.1],
            "XMax": [0.5],
            "YMin": [0.1],
            "YMax": [0.5],
        }
    )
    df.to_csv(annot_csv, index=False)

    output_path = tmp_path / "output" / "train.json"
    target_classes = ["Cat", "Dog"]  # Dog is index 1

    create_coco_json(
        split="train",
        metadata_dir=mock_metadata_dir,
        images_dir=mock_images_dir,
        output_path=output_path,
        target_classes=target_classes,
    )

    with open(output_path) as f:
        coco_data = json.load(f)

    # Dog should have category_id = 1
    assert coco_data["annotations"][0]["category_id"] == 1
    assert coco_data["categories"][1]["name"] == "Dog"
    assert coco_data["categories"][1]["id"] == 1

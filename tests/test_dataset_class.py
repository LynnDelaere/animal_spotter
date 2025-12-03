"""Tests for the AnimalSpotterDataset class."""

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from src.training.dataset_class import AnimalSpotterDataset


class MockProcessor:
    """Mock processor for testing purposes."""

    def __call__(
        self, images: Any, annotations: dict[str, Any], return_tensors: str
    ) -> dict[str, Any]:
        """Mock processing that returns dummy tensors."""
        # Create dummy pixel values
        pixel_values = torch.randn(3, 224, 224)

        # Create dummy labels
        labels = {
            "class_labels": torch.tensor(
                [ann["category_id"] for ann in annotations["annotations"]]
            ),
            "boxes": torch.tensor([ann["bbox"] for ann in annotations["annotations"]]),
        }

        return {
            "pixel_values": pixel_values.unsqueeze(0),  # Add batch dimension
            "labels": [labels],
        }


def create_mock_dataset(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Create a temporary dataset with mock data."""
    # Create image folder
    img_folder = tmp_path / "images"
    img_folder.mkdir()

    # Create dummy images
    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(img_folder / f"image_{i}.jpg")

    # Create COCO format annotation file
    coco_data = {
        "images": [
            {"id": 0, "file_name": "image_0.jpg", "width": 100, "height": 100},
            {"id": 1, "file_name": "image_1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "image_2.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 1,
                "image_id": 0,
                "category_id": 2,
                "bbox": [30, 30, 25, 25],
                "area": 625,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [5, 5, 15, 15],
                "area": 225,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 3,
                "bbox": [40, 40, 30, 30],
                "area": 900,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
            {"id": 3, "name": "bird"},
        ],
    }

    annotation_file = tmp_path / "annotations.json"
    with open(annotation_file, "w") as f:
        json.dump(coco_data, f)

    return img_folder, annotation_file, coco_data


def test_dataset_initialization(tmp_path: Path) -> None:
    """Test that the dataset initializes correctly."""
    img_folder, annotation_file, coco_data = create_mock_dataset(tmp_path)
    processor = MockProcessor()

    dataset = AnimalSpotterDataset(img_folder, annotation_file, processor)

    assert len(dataset) == 3
    assert len(dataset.img_ids) == 3
    assert len(dataset.images) == 3
    assert len(dataset.annotations) == 3


def test_dataset_length(tmp_path: Path) -> None:
    """Test that __len__ returns the correct number of images."""
    img_folder, annotation_file, _ = create_mock_dataset(tmp_path)
    processor = MockProcessor()

    dataset = AnimalSpotterDataset(img_folder, annotation_file, processor)

    assert len(dataset) == 3


def test_dataset_getitem(tmp_path: Path) -> None:
    """Test that __getitem__ returns the correct data structure."""
    img_folder, annotation_file, _ = create_mock_dataset(tmp_path)
    processor = MockProcessor()

    dataset = AnimalSpotterDataset(img_folder, annotation_file, processor)

    item = dataset[0]

    assert "pixel_values" in item
    assert "labels" in item
    assert isinstance(item["pixel_values"], torch.Tensor)
    assert isinstance(item["labels"], dict)


def test_dataset_annotations_mapping(tmp_path: Path) -> None:
    """Test that annotations are correctly mapped to images."""
    img_folder, annotation_file, coco_data = create_mock_dataset(tmp_path)
    processor = MockProcessor()

    dataset = AnimalSpotterDataset(img_folder, annotation_file, processor)

    # Image 0 should have 2 annotations
    assert len(dataset.annotations[0]) == 2

    # Image 1 should have 1 annotation
    assert len(dataset.annotations[1]) == 1

    # Image 2 should have 1 annotation
    assert len(dataset.annotations[2]) == 1


def test_dataset_loads_correct_image(tmp_path: Path) -> None:
    """Test that the correct image is loaded for each index."""
    img_folder, annotation_file, coco_data = create_mock_dataset(tmp_path)
    processor = MockProcessor()

    dataset = AnimalSpotterDataset(img_folder, annotation_file, processor)

    # Get the first item
    item = dataset[0]

    # Check that pixel_values has the correct shape (3 channels for RGB)
    assert item["pixel_values"].shape[0] == 3


def test_dataset_with_missing_annotations(tmp_path: Path) -> None:
    """Test dataset behavior when an image has no annotations."""
    img_folder, annotation_file, coco_data = create_mock_dataset(tmp_path)

    # Add an image without annotations
    coco_data["images"].append(
        {"id": 3, "file_name": "image_3.jpg", "width": 100, "height": 100}
    )

    # Create the image file
    img = Image.new("RGB", (100, 100), color=(150, 150, 150))
    img.save(img_folder / "image_3.jpg")

    # Update the annotation file
    with open(annotation_file, "w") as f:
        json.dump(coco_data, f)

    processor = MockProcessor()
    dataset = AnimalSpotterDataset(img_folder, annotation_file, processor)

    assert len(dataset) == 4
    # Image with id 3 should have no annotations
    assert dataset.annotations.get(3, []) == []

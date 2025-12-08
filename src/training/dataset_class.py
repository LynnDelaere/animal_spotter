"""Dataset class for managing animal spotter datasets."""

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from transformers import DetrImageProcessor


class AnimalSpotterDataset(Dataset):
    """Dataset class for animal spotter images and annotations.

    Uses a processor to handle image and annotation processing.
    """

    def __init__(
        self,
        img_folder: str | Path,
        annotation_file: str | Path,
        processor: DetrImageProcessor | Any,
    ) -> None:
        """Initialize the dataset.

        Args:
            img_folder (str or Path): Path to the folder containing images.
            annotation_file (str or Path): Path to the JSON file with annotations.
            processor: A processor object to handle image and annotation processing.
        """
        self.img_folder = Path(img_folder)
        self.processor = processor

        with open(annotation_file) as f:
            self.coco_data = json.load(f)

        # Create a mapping from image IDs to file names and annotations
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations: dict[int, list[dict]] = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.img_ids = list(self.images.keys())

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict:
        """Get a single item from the dataset by index."""
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        filename = img_info["file_name"]
        img_path = self.img_folder / filename

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get image size for bbox validation
        img_width, img_height = image.size

        # Get annotations for this image
        anns = self.annotations.get(img_id, [])

        # Process the boxes
        boxes = [ann["bbox"] for ann in anns]

        # Validate bounding boxes (COCO format: [x_min, y_min, width, height])
        problems = []
        for ann in anns:
            box = ann.get("bbox")
            if box is None or len(box) != 4:
                problems.append(
                    (img_id, img_path.name, box, "invalid format (expect 4 values)")
                )
                continue
            try:
                x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            except Exception:
                problems.append((img_id, img_path.name, box, "non-numeric values"))
                continue

            if w <= 0 or h <= 0:
                problems.append(
                    (img_id, img_path.name, box, "non-positive width/height")
                )
                continue
            if x < 0 or y < 0:
                problems.append((img_id, img_path.name, box, "negative x/y"))
                continue
            if x + w > img_width or y + h > img_height:
                problems.append(
                    (img_id, img_path.name, box, "bbox exceeds image bounds")
                )

        if problems:
            # Build a helpful error message so user can locate bad annotations
            details = "\n".join(
                [
                    f"image_id={p[0]} file={p[1]} bbox={p[2]} problem={p[3]}"
                    for p in problems
                ]
            )
            raise ValueError(
                f"Found invalid bounding box annotations for images in"
                f" {self.img_folder}:\n{details}\n"
                "Check your COCO annotations (bbox should be [x, y, width, height]"
                " with positive sizes)."
            )

        # Process the labels
        labels = [ann["category_id"] for ann in anns]

        # Process area
        areas = [ann["area"] for ann in anns]

        # Process iscrowd
        iscrowd = [ann["iscrowd"] for ann in anns]

        # Prepare target in COCO format for the processor
        encoding_target = {
            "image_id": img_id,
            "annotations": [
                {
                    "bbox": box,
                    "category_id": label,
                    "area": area,
                    "iscrowd": crowd,
                }
                for box, label, area, crowd in zip(
                    boxes, labels, areas, iscrowd, strict=False
                )
            ],
        }

        # Use the processor to process the image and target
        encoding = self.processor(
            images=image,
            annotations=encoding_target,
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze()  # Remove batch dimension

        target = encoding["labels"][0]  # Remove batch dimension

        return {
            "pixel_values": pixel_values,
            "labels": target,
        }

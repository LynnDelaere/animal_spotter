"""Evaluation script for the fine-tuned DETR model."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeAlias, cast

import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers.utils import TensorType

TargetSizesInput: TypeAlias = TensorType | list[tuple[Any, ...]]

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "processed"
TEST_IMAGES_DIR = IMAGES_DIR / "test"
TEST_ANNOTATIONS = LABELS_DIR / "test.json"
CLASSES_FILE = LABELS_DIR / "classes.yaml"
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "detr-finetuned"
FALLBACK_CHECKPOINT = "facebook/detr-resnet-50"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestImageDataset(Dataset):
    """Tiny helper dataset that only serves the test split."""

    def __init__(
        self,
        img_folder: str | Path,
        annotation_file: str | Path,
        processor: DetrImageProcessor,
    ) -> None:
        """Initialize the dataset with a COCO annotation file."""
        self.img_folder = Path(img_folder)
        self.processor = processor
        self.coco = COCO(annotation_file)
        self.img_ids = self.coco.getImgIds()

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return the encoded sample expected by the evaluation loop."""
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_folder / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        encoding = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "image_id": img_id,
            "target_size": (img_info["height"], img_info["width"]),
            "image_path": img_path,
        }


def collate_test_batch(batch: list[dict], processor: DetrImageProcessor) -> dict:
    """Pad different sized images so DETR can process the batch."""
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")

    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "image_ids": [item["image_id"] for item in batch],
        "target_sizes": torch.tensor(
            [item["target_size"] for item in batch],
            dtype=torch.float32,
        ),
        "image_paths": [item["image_path"] for item in batch],
    }


def load_classes(
    classes_file: Path = CLASSES_FILE,
) -> tuple[list[str], dict[int, str], dict[str, int]]:
    """Read classes.yaml so we can map between ids and labels."""
    with open(classes_file) as fp:
        classes: list[str] = yaml.safe_load(fp)

    id2label = {idx: label for idx, label in enumerate(classes)}
    label2id = {label: idx for idx, label in enumerate(classes)}
    return classes, id2label, label2id


def load_model(
    checkpoint: str | Path | None = None,
) -> tuple[DetrForObjectDetection, DetrImageProcessor, dict[int, str]]:
    """Load the fine-tuned checkpoint (or fall back to base DETR)."""
    classes, id2label, label2id = load_classes()

    if checkpoint is None:
        candidate = DEFAULT_MODEL_DIR
        checkpoint = candidate if candidate.exists() else FALLBACK_CHECKPOINT

    checkpoint = str(checkpoint)
    print(f"Loading model from {checkpoint}")

    processor = DetrImageProcessor.from_pretrained(checkpoint)
    model = DetrForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)
    model.eval()
    return model, processor, id2label


def _post_process_batch(  # noqa: PLR0913
    processor: DetrImageProcessor,
    outputs: Any,
    target_sizes: torch.Tensor,
    image_ids: list[int],
    score_threshold: float,
) -> list[dict]:
    detections = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=cast(TargetSizesInput, target_sizes),
    )

    preds: list[dict] = []
    for img_id, det in zip(image_ids, detections, strict=False):
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]
        for box, score, label in zip(boxes, scores, labels, strict=False):
            if score < score_threshold:
                continue
            x_min, y_min, x_max, y_max = box.tolist()
            preds.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "score": float(score),
                }
            )

    return preds


def run_test_evaluation(
    model: DetrForObjectDetection,
    processor: DetrImageProcessor,
    batch_size: int = 2,
    score_threshold: float = 0.1,
) -> tuple[dict[str, float], list[dict], COCO]:
    """Evaluate the test split and return COCO metrics and predictions."""
    dataset = TestImageDataset(TEST_IMAGES_DIR, TEST_ANNOTATIONS, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_test_batch(batch, processor),
    )

    predictions: list[dict] = []
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        pixel_mask = batch["pixel_mask"].to(DEVICE)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        batch_preds = _post_process_batch(
            processor=processor,
            outputs=outputs,
            target_sizes=batch["target_sizes"],
            image_ids=batch["image_ids"],
            score_threshold=score_threshold,
        )
        predictions.extend(batch_preds)

    coco_gt = dataset.coco
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics = {
            "mAP@[0.50:0.95]": float(coco_eval.stats[0]),
            "mAP@0.50": float(coco_eval.stats[1]),
            "mAP@0.75": float(coco_eval.stats[2]),
            "mAP_small": float(coco_eval.stats[3]),
            "mAP_medium": float(coco_eval.stats[4]),
            "mAP_large": float(coco_eval.stats[5]),
            "AR@1": float(coco_eval.stats[6]),
            "AR@10": float(coco_eval.stats[7]),
            "AR@100": float(coco_eval.stats[8]),
            "AR@100_small": float(coco_eval.stats[9]),
            "AR@100_medium": float(coco_eval.stats[10]),
            "AR@100_large": float(coco_eval.stats[11]),
        }
    else:
        print("No predictions above the score threshold.")
        metrics = {
            name: 0.0
            for name in [
                "mAP@[0.50:0.95]",
                "mAP@0.50",
                "mAP@0.75",
                "mAP_small",
                "mAP_medium",
                "mAP_large",
                "AR@1",
                "AR@10",
                "AR@100",
                "AR@100_small",
                "AR@100_medium",
                "AR@100_large",
            ]
        }

    return metrics, predictions, coco_gt


def get_visual_predictions(
    model: DetrForObjectDetection,
    processor: DetrImageProcessor,
    id2label: dict[int, str],
    image_paths: Iterable[str | Path],
    score_threshold: float = 0.5,
    max_detections: int = 25,
) -> list[dict]:
    """Return boxes/scores/labels per image so notebooks can draw plots."""
    visual_payloads: list[dict] = []
    for path in image_paths:
        path = Path(path)
        image = Image.open(path).convert("RGB")
        encoding = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(pixel_values=encoding["pixel_values"].to(DEVICE))

        target_sizes = torch.tensor([[image.height, image.width]], dtype=torch.float32)
        detections = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=cast(TargetSizesInput, target_sizes),
        )[0]

        boxes: list[list[float]] = []
        scores: list[float] = []
        labels: list[str] = []
        for box, score, label in zip(
            detections["boxes"],
            detections["scores"],
            detections["labels"],
            strict=False,
        ):
            if score < score_threshold:
                continue
            boxes.append(box.tolist())
            scores.append(float(score))
            labels.append(id2label[int(label)])
            if len(boxes) >= max_detections:
                break

        visual_payloads.append(
            {
                "image_path": path,
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        )

    return visual_payloads


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluating the DETR model."""
    parser = argparse.ArgumentParser(description="Evaluate DETR on the test set.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path or HF repo for the checkpoint (defaults to models/detr-finetuned).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="Ignore predictions below this confidence before metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point so the module can be executed as a script."""
    args = parse_args()
    model, processor, id2label = load_model(args.model_dir)

    metrics, predictions, _ = run_test_evaluation(
        model=model,
        processor=processor,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
    )

    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    example_images = list(TEST_IMAGES_DIR.glob("*.jpg"))[:3]
    if example_images:
        previews = get_visual_predictions(
            model=model,
            processor=processor,
            id2label=id2label,
            image_paths=example_images,
        )
        print("\nPrepared predictions for quick notebook previews:")
        for preview in previews:
            print(f"- {preview['image_path'].name}: {len(preview['boxes'])} detections")


if __name__ == "__main__":
    main()

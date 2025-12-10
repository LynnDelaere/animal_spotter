"""Tests for the DETR evaluation utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import src.evaluation.evaluate_detr as eval_mod
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


class DummyProcessor:
    """Processor stub that returns predictable tensors."""

    def __call__(
        self, images: Any, return_tensors: str = "pt"
    ) -> dict[str, torch.Tensor]:
        image_tensor = torch.ones(3, 16, 16)
        return {"pixel_values": image_tensor.unsqueeze(0)}


def create_coco_file(tmp_path: Path, image_name: str) -> Path:
    """Create a tiny COCO file with a single entry."""
    coco_payload = {
        "images": [
            {
                "id": 1,
                "file_name": image_name,
                "height": 12,
                "width": 34,
            }
        ],
        "annotations": [],
        "categories": [{"id": 0, "name": "animal"}],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(coco_payload))
    return json_path


def create_test_image(tmp_path: Path, image_name: str) -> Path:
    """Generate a dummy RGB image."""
    image_path = tmp_path / image_name
    Image.new("RGB", (34, 12), color="blue").save(image_path)
    return image_path


def test_test_image_dataset_returns_expected_payload(tmp_path: Path) -> None:
    """TestImageDataset should emit per-image metadata for DETR."""
    image_name = "sample.jpg"
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    create_test_image(img_dir, image_name)
    annotation_file = create_coco_file(tmp_path, image_name)
    processor = cast(DetrImageProcessor, DummyProcessor())

    dataset = eval_mod.TestImageDataset(img_dir, annotation_file, processor)

    item = dataset[0]
    assert item["image_id"] == 1
    assert item["target_size"] == (12, 34)
    assert item["image_path"].name == image_name
    assert item["pixel_values"].shape == (3, 16, 16)


class DummyPaddingProcessor(DummyProcessor):
    """Processor that can pad image tensors and do fake post-processing."""

    def pad(
        self, pixel_values: Iterable[torch.Tensor], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        tensors = list(pixel_values)
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        padded = []
        for tensor in tensors:
            pad_h = max_h - tensor.shape[1]
            pad_w = max_w - tensor.shape[2]
            padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
            padded.append(padded_tensor)

        stacked = torch.stack(padded)
        pixel_mask = torch.zeros((len(padded), max_h, max_w), dtype=torch.long)
        return {
            "pixel_values": stacked,
            "pixel_mask": pixel_mask,
        }


def test_collate_test_batch_pads_and_collects_metadata() -> None:
    """collate_test_batch should rely on the processor for padding."""
    processor = cast(DetrImageProcessor, DummyPaddingProcessor())
    batch = [
        {
            "pixel_values": torch.ones(3, 4, 4),
            "image_id": 5,
            "target_size": (4, 4),
            "image_path": Path("a.jpg"),
        },
        {
            "pixel_values": torch.ones(3, 2, 2),
            "image_id": 6,
            "target_size": (2, 2),
            "image_path": Path("b.jpg"),
        },
    ]

    collated = eval_mod.collate_test_batch(batch, processor)

    assert collated["pixel_values"].shape == (2, 3, 4, 4)
    assert torch.allclose(collated["pixel_values"][0], torch.ones(3, 4, 4))
    assert torch.allclose(collated["pixel_values"][1, :, :2, :2], torch.ones(3, 2, 2))
    assert torch.count_nonzero(collated["pixel_values"][1][:, 2:, :]) == 0
    assert collated["pixel_mask"].shape == (2, 4, 4)
    assert collated["image_ids"] == [5, 6]
    assert collated["target_sizes"].shape == (2, 2)


class DummyPostProcessor(DummyPaddingProcessor):
    """Processor stub to exercise post processing."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Any, torch.Tensor]] = []

    def post_process_object_detection(
        self,
        outputs: Any,
        target_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        self.calls.append((outputs, target_sizes))
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 20.0], [1.0, 1.0, 5.0, 5.0]]),
                "scores": torch.tensor([0.9, 0.05]),
                "labels": torch.tensor([1, 2]),
            },
            {
                "boxes": torch.tensor([[2.0, 3.0, 8.0, 11.0]]),
                "scores": torch.tensor([0.8]),
                "labels": torch.tensor([3]),
            },
        ]


def test_post_process_batch_filters_and_formats_predictions() -> None:
    """_post_process_batch should drop low scoring boxes and convert to xywh."""
    processor = cast(DetrImageProcessor, DummyPostProcessor())
    target_sizes = torch.tensor([[12, 34], [12, 34]], dtype=torch.float32)
    outputs = {"logits": torch.zeros(1)}

    predictions = eval_mod._post_process_batch(
        processor=processor,
        outputs=outputs,
        target_sizes=target_sizes,
        image_ids=[7, 9],
        score_threshold=0.5,
    )

    assert len(predictions) == 2
    assert predictions[0]["image_id"] == 7
    assert predictions[0]["bbox"] == [0.0, 0.0, 10.0, 20.0]
    assert predictions[1]["image_id"] == 9
    assert predictions[1]["bbox"] == [2.0, 3.0, 6.0, 8.0]
    assert all(pred["score"] >= 0.5 for pred in predictions)


class FakeDataset:
    """Replacement for TestImageDataset used in integration-style testing."""

    def __init__(self) -> None:
        self.samples = [
            {
                "pixel_values": torch.ones(3, 4, 4),
                "image_id": 11,
                "target_size": (4, 4),
                "image_path": Path("fake.jpg"),
            }
        ]
        self.coco = object()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class FakeDataLoader:
    """Small iterable that mimics torch.utils.data.DataLoader."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool,
        collate_fn: Any,
    ) -> None:
        del batch_size, shuffle
        self.batches = [collate_fn([dataset[0]])]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.batches)


class FakeModel:
    """Simple callable object mimicking a DETR model."""

    def __call__(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        assert pixel_values.shape[0] == pixel_mask.shape[0]
        return {"hidden_states": torch.zeros(1)}


class FakeEvalProcessor(DummyPostProcessor):
    """Processor ensuring run_test_evaluation sees no confident predictions."""

    def pad(
        self, pixel_values: Iterable[torch.Tensor], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        tensors = list(pixel_values)
        stacked = torch.stack(tensors)
        return {
            "pixel_values": stacked,
            "pixel_mask": torch.zeros_like(stacked),
        }

    def post_process_object_detection(
        self,
        outputs: Any,
        target_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        del outputs, target_sizes
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": torch.tensor([0.01]),
                "labels": torch.tensor([0]),
            }
        ]


def test_run_test_evaluation_handles_empty_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_test_evaluation should emit zeroed metrics when nothing passes threshold."""
    fake_dataset = FakeDataset()
    monkeypatch.setattr(
        eval_mod, "TestImageDataset", lambda *args, **kwargs: fake_dataset
    )
    monkeypatch.setattr(eval_mod, "DataLoader", FakeDataLoader)

    processor = cast(DetrImageProcessor, FakeEvalProcessor())
    model = cast(DetrForObjectDetection, FakeModel())

    metrics, predictions, coco_gt = eval_mod.run_test_evaluation(
        model=model,
        processor=processor,
        batch_size=1,
        score_threshold=0.5,
    )

    assert predictions == []
    expected_metric_names = [
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
    assert set(metrics.keys()) == set(expected_metric_names)
    assert all(value == 0.0 for value in metrics.values())
    assert coco_gt is fake_dataset.coco

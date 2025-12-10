"""Business logic helpers for running DETR inference."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError
from transformers import DetrForObjectDetection, DetrImageProcessor

from ..evaluation.evaluate_detr import load_model
from .schemas import Detection


class ModelService:
    """Lazy loader plus lightweight inference helper for DETR."""

    def __init__(
        self,
        checkpoint: str | None = None,
        score_threshold: float = 0.5,
        max_detections: int = 25,
    ) -> None:
        """Store baseline inference configuration and reset caches."""
        self.checkpoint = checkpoint
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self._model: DetrForObjectDetection | None = None
        self._processor: DetrImageProcessor | None = None
        self._id2label: dict[int, str] | None = None

    def _ensure_loaded(
        self,
    ) -> tuple[DetrForObjectDetection, DetrImageProcessor, dict[int, str]]:
        if self._model is None or self._processor is None or self._id2label is None:
            self._model, self._processor, self._id2label = load_model(self.checkpoint)
        return self._model, self._processor, self._id2label

    @property
    def is_ready(self) -> bool:
        """True once the HF model weights are already in memory."""
        return self._model is not None

    def predict(
        self,
        image_bytes: bytes,
        score_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> list[Detection]:
        """Run inference for a single image passed as raw bytes."""
        model, processor, id2label = self._ensure_loaded()
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file was not a valid image.") from exc

        encoding = processor(images=image, return_tensors="pt")
        device = next(model.parameters()).device
        with torch.no_grad():
            outputs = model(pixel_values=encoding["pixel_values"].to(device))

        post_processed = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=[(image.height, image.width)],
        )[0]

        detections: list[Detection] = []
        threshold = self.score_threshold if score_threshold is None else score_threshold
        limit = self.max_detections if max_detections is None else max_detections
        for box, score, label in zip(
            post_processed["boxes"],
            post_processed["scores"],
            post_processed["labels"],
            strict=False,
        ):
            if float(score) < threshold:
                continue
            detections.append(
                Detection(
                    box=box.tolist(),
                    score=float(score),
                    label=id2label[int(label)],
                )
            )
            if len(detections) >= limit:
                break

        return detections


@lru_cache(maxsize=1)
def get_model_service(
    checkpoint: str | None = None,
    score_threshold: float = 0.5,
    max_detections: int = 25,
) -> ModelService:
    """Return a memoized ModelService instance for FastAPI."""
    return ModelService(
        checkpoint=checkpoint,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

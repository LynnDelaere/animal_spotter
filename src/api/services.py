"""Business logic helpers for running DETR inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Annotated, Protocol

import torch
import yaml
from fastapi import Depends, HTTPException, Query
from PIL import Image, UnidentifiedImageError
from transformers import DetrForObjectDetection, DetrImageProcessor

from ..evaluation.evaluate_detr import load_model
from .schemas import Detection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CATALOG = PROJECT_ROOT / "configs" / "model_catalog.yaml"

ModelQuery = Annotated[
    str | None,
    Query(
        alias="model",
        description=(
            "Optional model key as returned by /api/models. "
            "Falls back to the default checkpoint when omitted."
        ),
    ),
]


@dataclass
class ModelConfigEntry:
    """Metadata describing a single deployable model variant."""

    key: str
    name: str
    checkpoint: str | None
    classes_file: Path | None
    description: str | None = None
    score_threshold: float = 0.5
    max_detections: int = 25


class ModelServiceProtocol(Protocol):
    """Contract describing the minimal interface for inference services."""

    def is_ready(self) -> bool:
        """True once the model weights live in memory."""
        ...

    def predict(
        self,
        image_bytes: bytes,
        score_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> list[Detection]:
        """Run inference and return structured detections."""
        ...

    def list_classes(self) -> list[str]:
        """Return the list of class labels."""
        ...


class ModelService(ModelServiceProtocol):
    """Lazy loader plus lightweight inference helper for DETR."""

    def __init__(
        self,
        checkpoint: str | None = None,
        classes_file: str | Path | None = None,
        score_threshold: float = 0.5,
        max_detections: int = 25,
    ) -> None:
        """Store baseline inference configuration and reset caches."""
        self.checkpoint = checkpoint
        self.classes_file = Path(classes_file) if classes_file else None
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self._model: DetrForObjectDetection | None = None
        self._processor: DetrImageProcessor | None = None
        self._id2label: dict[int, str] | None = None

    def _ensure_loaded(
        self,
    ) -> tuple[DetrForObjectDetection, DetrImageProcessor, dict[int, str]]:
        if self._model is None or self._processor is None or self._id2label is None:
            self._model, self._processor, self._id2label = load_model(
                self.checkpoint,
                self.classes_file,
            )
        return self._model, self._processor, self._id2label

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

    def list_classes(self) -> list[str]:
        """Return the ordered list of class labels the model can predict."""
        _, _, id2label = self._ensure_loaded()
        return [label for _, label in sorted(id2label.items())]


def get_model_service(
    checkpoint: str | None = None,
    classes_file: str | Path | None = None,
    score_threshold: float = 0.5,
    max_detections: int = 25,
) -> ModelServiceProtocol:
    """Return a memoized ModelService instance for FastAPI."""
    return ModelService(
        checkpoint=checkpoint,
        classes_file=classes_file,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )


class ModelRegistry:
    """Simple in-memory registry that lazy-loads ModelService instances."""

    def __init__(
        self,
        models: dict[str, ModelConfigEntry],
        default_key: str,
    ) -> None:
        """Initialize the registry with available models and the default slug."""
        if not models:
            raise ValueError("Cannot create a model registry without entries.")
        if default_key not in models:
            default_key = next(iter(models))
        self._models = models
        self._default_key = default_key
        self._services: dict[str, ModelServiceProtocol] = {}

    @property
    def default_key(self) -> str:
        """Return the slug for the model used when no key is provided."""
        return self._default_key

    def list(self) -> list[ModelConfigEntry]:
        """Return metadata for every registered model."""
        return list(self._models.values())

    def get_config(self, key: str | None) -> ModelConfigEntry:
        """Return the config entry for a given slug (or the default)."""
        slug = self._resolve_key(key)
        return self._models[slug]

    def get(self, key: str | None) -> ModelServiceProtocol:
        """Return (and cache) the ModelService for the requested slug."""
        slug = self._resolve_key(key)
        if slug not in self._services:
            config = self._models[slug]
            self._services[slug] = ModelService(
                checkpoint=config.checkpoint,
                classes_file=config.classes_file,
                score_threshold=config.score_threshold,
                max_detections=config.max_detections,
            )
        return self._services[slug]

    def _resolve_key(self, requested: str | None) -> str:
        if requested is None:
            return self._default_key
        if requested not in self._models:
            raise KeyError(requested)
        return requested


def _resolve_checkpoint_path(value: str | Path | None) -> str | None:
    """Return either an absolute path or pass through HF repo ids."""
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return str(candidate)
    return str(value)


def _resolve_classes_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_catalog_from_mapping(
    data: dict,
) -> tuple[dict[str, ModelConfigEntry], str]:
    models_config = data.get("models") or {}
    models: dict[str, ModelConfigEntry] = {}
    for key, raw in models_config.items():
        if not isinstance(raw, dict):
            continue
        models[key] = ModelConfigEntry(
            key=key,
            name=str(raw.get("display_name", key)),
            description=raw.get("description"),
            checkpoint=_resolve_checkpoint_path(raw.get("checkpoint")),
            classes_file=_resolve_classes_path(raw.get("classes_file")),
            score_threshold=float(raw.get("score_threshold", 0.5)),
            max_detections=int(raw.get("max_detections", 25)),
        )
    if not models:
        raise ValueError("Model catalog did not define any entries.")
    default_key = str(data.get("default_model") or next(iter(models)))
    if default_key not in models:
        raise ValueError(
            f"Default model '{default_key}' missing from model catalog entries."
        )
    return models, default_key


def _catalog_from_env() -> tuple[dict[str, ModelConfigEntry], str]:
    checkpoint = _resolve_checkpoint_path(os.getenv("MODEL_DIR"))
    entry = ModelConfigEntry(
        key="default",
        name="Default model",
        description="Auto-generated entry based on MODEL_DIR.",
        checkpoint=checkpoint,
        classes_file=_resolve_classes_path(os.getenv("MODEL_CLASSES_FILE")),
    )
    return {"default": entry}, "default"


def load_model_registry(
    catalog_path: str | Path | None = None,
) -> ModelRegistry:
    """Materialize a registry either from a YAML catalog or env vars."""
    candidates: list[str | Path | None] = [
        catalog_path,
        os.getenv("MODEL_CATALOG"),
        DEFAULT_MODEL_CATALOG,
    ]
    config_path: Path | None = None
    for candidate in candidates:
        if candidate is None or candidate == "":
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            config_path = path
            break
    if config_path:
        with open(config_path) as fp:
            raw = yaml.safe_load(fp) or {}
        if not isinstance(raw, dict):
            raise ValueError("Model catalog must be a mapping.")
        models, default_key = _load_catalog_from_mapping(raw)
    else:
        models, default_key = _catalog_from_env()
    return ModelRegistry(models=models, default_key=default_key)


def get_model_registry() -> ModelRegistry:
    """Cached accessor used by FastAPI dependencies."""
    return load_model_registry()


def provide_model_service(
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    model_name: ModelQuery = None,
) -> ModelServiceProtocol:
    """Dependency wrapper selecting the requested model (or the default)."""
    try:
        return registry.get(model_name)
    except KeyError as exc:  # pragma: no cover - FastAPI handles response
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_name}'.",
        ) from exc

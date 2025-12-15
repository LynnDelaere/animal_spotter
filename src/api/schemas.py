"""Shared Pydantic schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Represents a single object detection prediction."""

    box: list[float] = Field(
        ...,
        description="[x_min, y_min, x_max, y_max] in pixel coordinates.",
        min_length=4,
        max_length=4,
    )
    score: float = Field(..., ge=0.0, le=1.0)
    label: str


class PredictionResponse(BaseModel):
    """Payload returned by /predictions."""

    filename: str
    detections: list[Detection]


class HealthResponse(BaseModel):
    """Basic heartbeat response."""

    status: str
    model_loaded: bool


class ClassesResponse(BaseModel):
    """List of class labels the model can predict."""

    classes: list[str]


class ModelInfo(BaseModel):
    """Metadata describing a deployed model."""

    slug: str = Field(..., description="Identifier passed as the `model` query.")
    name: str
    description: str | None = None


class ModelsResponse(BaseModel):
    """Payload describing all models available on the backend."""

    default_model: str
    models: list[ModelInfo]

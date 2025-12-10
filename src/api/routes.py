"""API route definitions."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from .schemas import HealthResponse, PredictionResponse
from .services import ModelService, get_model_service

api_router = APIRouter(
    prefix="/api",
    tags=["detr"],
)


@api_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Simple readiness probe.",
)
async def health(
    service: Annotated[ModelService, Depends(get_model_service)],
) -> HealthResponse:
    """Return a tiny heartbeat payload for uptime trackers."""
    return HealthResponse(status="ok", model_loaded=service.is_ready)


@api_router.post(
    "/predictions",
    response_model=PredictionResponse,
    summary="Run inference for a single image.",
    status_code=status.HTTP_200_OK,
)
async def run_prediction(
    file: Annotated[UploadFile, File(...)],
    service: Annotated[ModelService, Depends(get_model_service)],
    score_threshold: Annotated[float, Query(0.5, ge=0.0, le=1.0)],
    max_detections: Annotated[int, Query(25, ge=1, le=100)],
) -> PredictionResponse:
    """Accept an image upload and run the DETR model inference."""
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file was empty.",
        )

    try:
        detections = await run_in_threadpool(
            service.predict,
            contents,
            score_threshold,
            max_detections,
        )
    except ValueError as exc:  # Pillow failed to read the bytes
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    return PredictionResponse(filename=file.filename or "upload", detections=detections)

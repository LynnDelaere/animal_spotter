"""Application factory for the Animal Spotter REST API."""

from fastapi import FastAPI

from .routes import api_router


def create_app() -> FastAPI:
    """Return a configured FastAPI application instance."""
    app = FastAPI(
        title="Animal Spotter API",
        version="0.1.0",
        description="REST API skeleton for DETR model inference and health checks.",
    )
    app.include_router(api_router)
    return app


app = create_app()

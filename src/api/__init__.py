"""Server package exposing the REST API entrypoints."""

from .app import create_app

__all__ = ["create_app"]

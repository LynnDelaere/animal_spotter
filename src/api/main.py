"""Module to launch the FastAPI server via `python -m src.api.main`."""

import os

import uvicorn


def main() -> None:
    """Start the ASGI server with sensible local defaults."""
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    reload_enabled = os.environ.get("API_RELOAD", "0") == "1"
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()

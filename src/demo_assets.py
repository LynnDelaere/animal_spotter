"""Common asset paths shared between the Gradio app and deployment helpers."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
TEST_IMAGES_DIR = DATA_DIR / "images" / "test"

# Curated images preloaded in the Gradio UI (and copied for HF Spaces).
EXAMPLE_IMAGE_PATHS = [
    TEST_IMAGES_DIR / "0a389d85258b5e0a.jpg",
    TEST_IMAGES_DIR / "e78dadb8f6a68e72.jpg",
    TEST_IMAGES_DIR / "e6cb3db8bffb3251.jpg",
    TEST_IMAGES_DIR / "161a28fbe1153fe2.jpg",
    DATA_DIR / "PXL_20251019_120353831.jpg",
]

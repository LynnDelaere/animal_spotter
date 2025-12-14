#!/usr/bin/env python3
"""Create a lightweight folder that can be deployed to Hugging Face Spaces."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "build" / "hf_space"

# Only the files below are required to run the Gradio UI remotely.
INCLUDE_PATHS = [
    "src",
    "pyproject.toml",
]

CLASSES_FILE = PROJECT_ROOT / "data" / "processed" / "classes.yaml"
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "images" / "test"
SPACE_REQUIREMENTS = PROJECT_ROOT / "requirements.space.txt"
FALLBACK_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
SPACE_README_TEMPLATE = PROJECT_ROOT / "README.space.md"
DEFAULT_README = PROJECT_ROOT / "README.md"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "detr-finetuned"
ESSENTIAL_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)


def _copy_path(src: Path, dest: Path) -> None:
    if src.is_dir():
        shutil.copytree(
            src,
            dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.py[co]"),
        )
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _copy_example_images(dest: Path, limit: int) -> int:
    """Copy a handful of test images so the Examples widget keeps working."""
    if not TEST_IMAGES_DIR.exists():
        return 0

    copied = 0
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(TEST_IMAGES_DIR.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        shutil.copy2(path, dest / path.name)
        copied += 1
        if copied >= limit:
            break
    if copied == 0:
        dest.rmdir()
    return copied


def _copy_checkpoint_assets(src_dir: Path, dest_dir: Path) -> None:
    missing: list[str] = []
    copied_any = False
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename in ESSENTIAL_MODEL_FILES:
        src_file = src_dir / filename
        if not src_file.exists():
            missing.append(filename)
            continue
        _copy_path(src_file, dest_dir / filename)
        copied_any = True
    if missing:
        print(f"[warn] Missing checkpoint files in {src_dir}: {', '.join(missing)}")
    if not copied_any:
        dest_dir.rmdir()


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for the Space packaging helper."""
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a trimmed-down folder for `gradio deploy` so large "
            "artifacts are left out of the Hugging Face Space."
        )
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Target folder (default: build/hf_space).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Path to the fine-tuned checkpoint directory.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=4,
        help="Number of test images to copy for Gradio's Examples widget.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for staging a lightweight folder for deployment."""
    args = parse_args()
    dest_root: Path = args.dest
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    for relative in INCLUDE_PATHS:
        src = PROJECT_ROOT / relative
        if not src.exists():
            print(f"[skip] {relative} is missing.")
            continue
        print(f"[copy] {relative}")
        _copy_path(src, dest_root / relative)

    if CLASSES_FILE.exists():
        rel = Path("data") / "processed" / CLASSES_FILE.name
        print("[copy] data/processed/classes.yaml")
        _copy_path(CLASSES_FILE, dest_root / rel)
    else:
        print("[warn] classes.yaml not found; remote app will use default labels.")

    if SPACE_README_TEMPLATE.exists():
        print("[copy] README.space.md -> README.md")
        _copy_path(SPACE_README_TEMPLATE, dest_root / "README.md")
    elif DEFAULT_README.exists():
        print("[copy] README.md")
        _copy_path(DEFAULT_README, dest_root / "README.md")
    else:
        print("[warn] README template missing; no README copied.")

    req_src = (
        SPACE_REQUIREMENTS if SPACE_REQUIREMENTS.exists() else FALLBACK_REQUIREMENTS
    )
    print(f"[copy] {req_src.name} -> requirements.txt")
    _copy_path(req_src, dest_root / "requirements.txt")

    checkpoint_dir = args.checkpoint
    if checkpoint_dir.exists():
        target_dir = dest_root / "models" / checkpoint_dir.name
        print(f"[copy] Minimal checkpoint assets from {checkpoint_dir} -> {target_dir}")
        _copy_checkpoint_assets(checkpoint_dir, target_dir)
    else:
        print(f"[info] Checkpoint directory {checkpoint_dir} not found; skipping.")

    copied = _copy_example_images(
        dest_root / "data" / "images" / "test",
        limit=args.example_limit,
    )
    if copied:
        print(f"[copy] {copied} example image(s) for the demo.")
    else:
        print("[info] No sample images copied (folder missing?).")

    print(f"\nReady to deploy from: {dest_root}")
    print(
        f"Run `cd {dest_root} && source ../../.venv/bin/activate && "
        "gradio deploy --title animal-spotter --app-file src/gradio_app.py`"
    )


if __name__ == "__main__":
    main()

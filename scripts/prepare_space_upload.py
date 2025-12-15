#!/usr/bin/env python3
"""Create a lightweight folder that can be deployed to Hugging Face Spaces."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.demo_assets import EXAMPLE_IMAGE_PATHS  # noqa: E402

DEFAULT_DEST = PROJECT_ROOT / "build" / "hf_space"
DEFAULT_MODEL_CATALOG = PROJECT_ROOT / "configs" / "model_catalog.yaml"

# Only the files below are required to run the Gradio UI remotely.
INCLUDE_PATHS = [
    "src",
    "pyproject.toml",
]

CLASSES_FILE = PROJECT_ROOT / "data" / "processed" / "classes.yaml"
SPACE_REQUIREMENTS = PROJECT_ROOT / "requirements.space.txt"
FALLBACK_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
SPACE_README_TEMPLATE = PROJECT_ROOT / "README.space.md"
DEFAULT_README = PROJECT_ROOT / "README.md"
ESSENTIAL_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)
DEFAULT_EXAMPLE_LIMIT = len(EXAMPLE_IMAGE_PATHS)


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


def _copy_example_images(dest_root: Path, limit: int | None) -> int:
    """Copy the curated demo images so the Examples widget keeps working."""
    candidates = EXAMPLE_IMAGE_PATHS if limit is None else EXAMPLE_IMAGE_PATHS[:limit]
    copied = 0
    for relative_path in candidates:
        src = (
            relative_path
            if relative_path.is_absolute()
            else PROJECT_ROOT / relative_path
        )
        if not src.exists():
            print(f"[warn] Example image missing locally: {src}")
            continue
        target = dest_root / relative_path
        _copy_path(src, target)
        copied += 1
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
        default=None,
        help="Optional extra checkpoint directory to copy.",
    )
    parser.add_argument(
        "--model-catalog",
        type=Path,
        default=DEFAULT_MODEL_CATALOG,
        help="Path to the YAML catalog describing deployed models.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=DEFAULT_EXAMPLE_LIMIT,
        help="Number of curated example images to copy for Gradio's Examples widget.",
    )
    return parser.parse_args()


def _resolve_existing_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = PROJECT_ROOT / path if not path.is_absolute() else path
    return candidate if candidate.exists() else None


def _relative_to_project(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return Path("models") / path.name


def _parse_model_catalog(path: Path | None) -> tuple[list[Path], list[Path]]:
    if path is None or not path.exists():
        return [], []
    with open(path) as fp:
        raw = yaml.safe_load(fp) or {}
    if not isinstance(raw, dict):
        return [], []
    models = raw.get("models") or {}
    checkpoints: list[Path] = []
    class_files: list[Path] = []
    for config in models.values():
        if not isinstance(config, dict):
            continue
        checkpoint_path = None
        checkpoint_entry = config.get("checkpoint")
        if checkpoint_entry:
            checkpoint_path = _resolve_existing_path(Path(str(checkpoint_entry)))
        if checkpoint_path:
            checkpoints.append(checkpoint_path)
        classes_entry = config.get("classes_file")
        if classes_entry:
            class_path = _resolve_existing_path(Path(str(classes_entry)))
            if class_path:
                class_files.append(class_path)
    return checkpoints, class_files


def _prepare_destination(dest_root: Path) -> None:
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)


def _copy_project_includes(dest_root: Path) -> None:
    for relative in INCLUDE_PATHS:
        src = PROJECT_ROOT / relative
        if not src.exists():
            print(f"[skip] {relative} is missing.")
            continue
        print(f"[copy] {relative}")
        _copy_path(src, dest_root / relative)


def _copy_readme_and_requirements(dest_root: Path) -> None:
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


def _copy_catalog_file(dest_root: Path, catalog_path: Path | None) -> None:
    if catalog_path and catalog_path.exists():
        rel = _relative_to_project(catalog_path)
        print(f"[copy] {rel}")
        _copy_path(catalog_path, dest_root / rel)
    elif catalog_path:
        print(f"[warn] Model catalog {catalog_path} not found; skipping.")


def _copy_checkpoint_dirs(dest_root: Path, checkpoint_dirs: list[Path]) -> None:
    copied_any_checkpoint = False
    for src_dir in {path.resolve() for path in checkpoint_dirs if path.exists()}:
        rel_dir = _relative_to_project(src_dir)
        target_dir = dest_root / rel_dir
        print(f"[copy] Minimal checkpoint assets from {src_dir} -> {target_dir}")
        _copy_checkpoint_assets(src_dir, target_dir)
        copied_any_checkpoint = True
    if not copied_any_checkpoint:
        print("[info] No checkpoint directories copied (local files missing?).")


def _copy_class_files(dest_root: Path, class_files: list[Path]) -> None:
    class_candidates: list[Path] = []
    if CLASSES_FILE.exists():
        class_candidates.append(CLASSES_FILE)
    class_candidates.extend(path for path in class_files if path.exists())

    copied_class_files = False
    for class_file in {path.resolve() for path in class_candidates}:
        rel_target = _relative_to_project(class_file)
        print(f"[copy] {rel_target}")
        _copy_path(class_file, dest_root / rel_target)
        copied_class_files = True

    if not copied_class_files:
        print("[warn] No classes.yaml files found; remote app will use default labels.")


def _copy_examples_for_space(dest_root: Path, limit: int | None) -> None:
    copied = _copy_example_images(dest_root, limit=limit)
    if copied:
        print(f"[copy] {copied} curated example image(s) for the demo.")
    else:
        print("[info] No curated sample images copied (files missing?).")


def _normalize_catalog_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _collect_checkpoint_dirs(
    extra_checkpoint: Path | None, catalog_checkpoint_dirs: list[Path]
) -> list[Path]:
    checkpoint_dirs: list[Path] = []
    resolved_extra = _resolve_existing_path(extra_checkpoint)
    if resolved_extra:
        checkpoint_dirs.append(resolved_extra)
    checkpoint_dirs.extend(path for path in catalog_checkpoint_dirs if path.exists())
    return checkpoint_dirs


def main() -> None:
    """Entry point for staging a lightweight folder for deployment."""
    args = parse_args()
    dest_root: Path = args.dest
    _prepare_destination(dest_root)
    _copy_project_includes(dest_root)

    catalog_path = _normalize_catalog_path(args.model_catalog)
    catalog_checkpoint_dirs, catalog_class_files = _parse_model_catalog(catalog_path)
    _copy_class_files(dest_root, catalog_class_files)
    _copy_readme_and_requirements(dest_root)
    _copy_catalog_file(dest_root, catalog_path)

    checkpoint_dirs = _collect_checkpoint_dirs(args.checkpoint, catalog_checkpoint_dirs)
    _copy_checkpoint_dirs(dest_root, checkpoint_dirs)
    _copy_examples_for_space(dest_root, limit=args.example_limit)

    print(f"\nReady to deploy from: {dest_root}")
    print(
        f"Run `cd {dest_root} && source ../../.venv/bin/activate && "
        "gradio deploy --title animal-spotter --app-file src/gradio_app.py`"
    )


if __name__ == "__main__":
    main()

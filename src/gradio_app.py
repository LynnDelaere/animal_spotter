"""Interactive Gradio demo around the Animal Spotter DETR model."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import cast

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

if __package__ is None or __package__ == "":
    # Allow running as `python src/gradio_app.py` by ensuring src/ is on sys.path.
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_ROOT))
    from src.api.schemas import Detection
    from src.api.services import (
        ModelConfigEntry,
        ModelRegistry,
        load_model_registry,
    )
    from src.demo_assets import EXAMPLE_IMAGE_PATHS
else:  # pragma: no cover - exercised when executed as a module
    from .api.schemas import Detection
    from .api.services import (
        ModelConfigEntry,
        ModelRegistry,
        load_model_registry,
    )
    from .demo_assets import EXAMPLE_IMAGE_PATHS


def _image_to_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL image into PNG bytes for the inference service."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_registry_from_args(
    checkpoint: str | None,
    catalog: str | Path | None,
    score_threshold: float,
    max_detections: int,
) -> ModelRegistry:
    """Return either a single-model registry or load the YAML catalog."""
    if checkpoint:
        entry = ModelConfigEntry(
            key="custom",
            name="Custom checkpoint",
            description="Loaded via --checkpoint CLI flag.",
            checkpoint=checkpoint,
            classes_file=None,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        return ModelRegistry(models={"custom": entry}, default_key="custom")
    return load_model_registry(catalog)


def _format_for_display(
    detections: list[Detection],
) -> tuple[list[list[float | str]], list[tuple[int, int, int, int, str]]]:
    """Convert detections to raw table rows plus drawing instructions."""
    table_rows: list[list[float | str]] = []
    boxes: list[tuple[int, int, int, int, str]] = []
    for det in detections:
        x_min, y_min, x_max, y_max = det.box
        boxes.append(
            (
                int(round(x_min)),
                int(round(y_min)),
                int(round(x_max)),
                int(round(y_max)),
                f"{det.label} ({det.score:.2f})",
            )
        )
        table_rows.append(
            [
                det.label,
                round(det.score, 3),
                round(x_min, 1),
                round(y_min, 1),
                round(x_max, 1),
                round(y_max, 1),
            ]
        )
    return table_rows, boxes


def _draw_boxes(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int, str]],
) -> Image.Image:
    """Return a copy of the image with notebook-style DETR bounding boxes."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=20)
    width, height = annotated.width, annotated.height
    for x_min, y_min, x_max, y_max, label in boxes:
        x1 = max(0, min(x_min, width - 1))
        y1 = max(0, min(y_min, height - 1))
        x2 = max(0, min(x_max, width))
        y2 = max(0, min(y_max, height))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)

        draw.rectangle((x1, y1, x2, y2), outline="#07f5e5", width=5)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        padding = 8
        bg_left = max(0, min(x1, width - 1))
        bg_right = min(width, bg_left + text_w + 2 * padding)
        bg_top = max(0, min(y1 - text_h - 2 * padding, height - 1))
        bg_bottom = min(height, max(y1, bg_top + text_h + 2 * padding))
        draw.rectangle((bg_left, bg_top, bg_right, bg_bottom), fill="#111827")
        draw.text(
            (bg_left + padding, bg_top + padding),
            label,
            font=font,
            fill="#fef3c7",
        )
    return annotated


def build_predict_fn(
    registry: ModelRegistry,
) -> Callable[[Image.Image | None, str | None, float, int], tuple[Image.Image, list]]:
    """Wrap the service predict method so Gradio can call it."""

    def _predict(
        image: Image.Image | None,
        model_key: str | None,
        score_threshold: float,
        max_detections: int,
    ) -> tuple[Image.Image, list]:
        if image is None:
            raise gr.Error("Upload (or pick) an image first.")

        try:
            service = registry.get(model_key)
        except KeyError as exc:  # pragma: no cover - user interaction
            raise gr.Error(
                f"Unknown model '{model_key}'. Pick another option."
            ) from exc

        detections = service.predict(
            _image_to_bytes(image),
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        table, boxes = _format_for_display(detections)
        annotated = _draw_boxes(image, boxes)
        return annotated, table

    return _predict


def _discover_examples(limit: int | None = None) -> list[Path]:
    """Return the curated local images for the Examples widget."""
    candidates = EXAMPLE_IMAGE_PATHS if limit is None else EXAMPLE_IMAGE_PATHS[:limit]
    return [path for path in candidates if path.exists()]


def _model_info_with_classes(
    registry: ModelRegistry, model_key: str | None
) -> tuple[ModelConfigEntry | None, list[str]]:
    """Return the model config plus its class labels (if available)."""
    try:
        config = registry.get_config(model_key)
        service = registry.get(config.key)
    except KeyError:
        return None, []
    return config, service.list_classes()


def _classes_for_model(registry: ModelRegistry, model_key: str | None) -> str:
    """Return the Markdown snippet describing classes for a given slug."""
    config, classes = _model_info_with_classes(registry, model_key)
    if not config:
        return "No model selected."
    if not classes:
        return "No class labels available for this model."
    md_lines = [
        f"**Model '{config.name}' classes ({len(classes)} total):**",
        "",
        "```",
    ]
    md_lines.extend(f"- {label}" for label in classes)
    md_lines.append("```")
    return "\n".join(md_lines)


def build_interface(
    registry: ModelRegistry,
) -> gr.Blocks:
    """Construct the Blocks layout with inputs, controls and results."""
    predict_fn = build_predict_fn(registry)
    examples = _discover_examples()
    default_config = registry.get_config(None)
    default_key = default_config.key
    default_score = default_config.score_threshold
    default_limit = default_config.max_detections
    classes_md = _classes_for_model(registry, default_key)
    model_entries = registry.list()
    model_choices = [entry.key for entry in model_entries]
    models_summary = " | ".join(
        f"{entry.name} ({entry.key})" for entry in model_entries
    )

    def _update_class_widgets(selected_key: str | None) -> str:
        return _classes_for_model(registry, selected_key)

    with gr.Blocks(title="Animal Spotter Demo") as demo:
        gr.Markdown(
            "# 🐾 Animal Spotter DEMO\n"
            "Upload an image (or pick one below) to run the DETR detector and "
            "visualize the predicted bounding boxes."
        )
        model_dropdown = gr.Dropdown(
            label="Model checkpoint",
            info=models_summary or None,
            choices=model_choices,
            value=default_key,
        )
        with gr.Accordion("Classes", open=False):
            classes_panel = gr.Markdown(classes_md)

        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(
                    label="Input image",
                    type="pil",
                    height=480,
                )
                score_slider = gr.Slider(
                    label="Score threshold",
                    minimum=0.05,
                    maximum=0.95,
                    step=0.05,
                    value=default_score,
                )
                limit_slider = gr.Slider(
                    label="Max detections",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=default_limit,
                )
                run_btn = gr.Button("Run detection", variant="primary")
                if examples:
                    gr.Examples(
                        label="Sample wildlife photos",
                        examples=examples,
                        inputs=[image_input],
                    )
            with gr.Column(scale=3):
                annotated_output = gr.Image(
                    label="Predictions",
                    height=480,
                    type="pil",
                )
                table_output = gr.DataFrame(
                    label="Raw detections",
                    wrap=True,
                    type="array",
                    datatype=["str", "number", "number", "number", "number", "number"],
                    headers=["label", "score", "x_min", "y_min", "x_max", "y_max"],
                )

        run_btn.click(
            predict_fn,
            inputs=[image_input, model_dropdown, score_slider, limit_slider],
            outputs=[annotated_output, table_output],
        )
        model_dropdown.change(
            _update_class_widgets,
            inputs=model_dropdown,
            outputs=classes_panel,
        )

    return cast(gr.Blocks, demo)


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for configuring the Gradio server."""
    parser = argparse.ArgumentParser(
        description="Launch the Animal Spotter Gradio demo."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional local path or HF repo id for the DETR checkpoint.",
    )
    parser.add_argument(
        "--model-catalog",
        type=str,
        default=None,
        help=(
            "Path to a YAML file describing available checkpoints "
            "(defaults to configs/model_catalog.yaml when present)."
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Default score threshold used when the app loads.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=25,
        help="Default limit for the number of detections to display.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host passed to Gradio (defaults to 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port passed to Gradio (defaults to 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio's public sharing tunnel.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point to initialize the model service and launch Gradio."""
    args = parse_args()
    registry = _build_registry_from_args(
        checkpoint=args.checkpoint,
        catalog=args.model_catalog,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
    )
    app = build_interface(registry)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

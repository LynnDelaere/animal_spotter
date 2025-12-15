"""Training script for fine-tuning DETR model on animal spotter dataset."""

import argparse
from pathlib import Path

import torch
import yaml
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .dataset_class import AnimalSpotterDataset

# Configure paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "processed"
LOG_DIR = ROOT_DIR / "logs"
CONFIG_DIR = ROOT_DIR / "configs"

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def resolve_path(path_value: str | None, default: Path) -> Path:
    """Return an absolute path based on config input or fallback default."""
    if not path_value:
        return default
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def ensure_exists(path: Path, description: str) -> Path:
    """Ensure a path exists and return it, otherwise throw a helpful error."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")
    return path


# Collate function to handle variable image size
def collate_fn(batch: list[dict], processor: DetrImageProcessor) -> dict:
    """Custom collate function to handle batches of images with different sizes.

    DETR needs a custom collate function to handle batches of images with
    different sizes. The processor will take care of padding.
    """
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    encoding = processor.pad(pixel_values, return_tensors="pt")

    batch_dict = {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }
    return batch_dict


# Load processor and model
def main() -> None:
    """Main training function to fine-tune DETR model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train DETR model")
    parser.add_argument(
        "--config",
        type=str,
        default="dataset_v2.yaml",
        help="Path to YAML config file (default: dataset_v2.yaml)",
    )
    args = parser.parse_args()

    # Load config to get version (support absolute or relative paths)
    config_arg = Path(args.config)
    config_candidates = (
        [config_arg]
        if config_arg.is_absolute()
        else [ROOT_DIR / config_arg, CONFIG_DIR / config_arg]
    )

    for path in config_candidates:
        if path.exists():
            config_path = path
            break
    else:
        raise FileNotFoundError(f"Config file not found. Tried: {config_candidates}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    version = config.get("dataset", {}).get("version", "v1")

    paths_config = config.get("paths", {})
    data_dir = resolve_path(paths_config.get("raw_dir"), DATA_DIR)
    images_base_dir = resolve_path(paths_config.get("images_dir"), data_dir / "images")
    processed_dir = resolve_path(
        paths_config.get("processed_dir"), data_dir / "processed"
    )

    versioned_processed_dir = processed_dir / version
    if versioned_processed_dir.exists():
        labels_dir = versioned_processed_dir
    else:
        labels_dir = processed_dir
    ensure_exists(labels_dir, "Processed dataset directory")

    versioned_images_dir = images_base_dir / version
    if versioned_images_dir.exists():
        images_dir = versioned_images_dir
    else:
        images_dir = images_base_dir
    ensure_exists(images_dir, "Images base directory")

    # Create versioned model and log directories
    model_base_dir = ROOT_DIR / "models" / "detr-finetuned"
    model_version_dir = model_base_dir / version
    model_version_dir.mkdir(parents=True, exist_ok=True)

    log_version_dir = LOG_DIR / version
    log_version_dir.mkdir(parents=True, exist_ok=True)

    classes_path = labels_dir / "classes.yaml"
    if classes_path.exists():
        with open(classes_path) as f:
            classes = yaml.safe_load(f)
    else:
        classes = config.get("dataset", {}).get("classes")
        if not classes:
            raise FileNotFoundError(
                "Could not locate classes.yaml and no classes defined in config."
            )

    id2label = {i: label for i, label in enumerate(classes)}
    label2id = {label: i for i, label in enumerate(classes)}

    print(f"Training model with {len(classes)} classes. Classes: {classes}")

    checkpoint = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(checkpoint)

    train_images_dir = ensure_exists(images_dir / "train", "Training images directory")
    val_images_dir = ensure_exists(
        images_dir / "validation", "Validation images directory"
    )
    train_annotations = ensure_exists(labels_dir / "train.json", "Training annotations")
    val_annotations = ensure_exists(
        labels_dir / "validation.json", "Validation annotations"
    )

    train_dataset = AnimalSpotterDataset(
        img_folder=train_images_dir,
        annotation_file=train_annotations,
        processor=processor,
    )

    val_dataset = AnimalSpotterDataset(
        img_folder=val_images_dir,
        annotation_file=val_annotations,
        processor=processor,
    )

    model = DetrForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(model_version_dir),
        per_device_train_batch_size=4,
        num_train_epochs=50,
        fp16=torch.cuda.is_available(),
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=1e-4,
        logging_dir=str(log_version_dir),
        logging_strategy="steps",
        logging_steps=50,
        report_to=["tensorboard"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    # Use a lambda to pass processor to collate_fn
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=lambda batch: collate_fn(batch, processor),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("Starting training...")
    print(f"Monitor training via TensorBoard at {log_version_dir}")
    print(f"Model version: {version}")
    print(f"Config: {config_path}")

    trainer.train()

    # Save the final model
    trainer.save_model()
    # Save processor to the same output directory
    processor.save_pretrained(str(model_version_dir))
    print(f"Model saved to {model_version_dir}")


if __name__ == "__main__":
    main()

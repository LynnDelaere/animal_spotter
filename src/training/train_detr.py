"""Training script for fine-tuning DETR model on animal spotter dataset."""

import argparse
from pathlib import Path

import torch
import yaml
from dataset_class import AnimalSpotterDataset
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

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

    # Load config to get version
    config_path = CONFIG_DIR / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    version = config.get("dataset", {}).get("version", "v1")

    # Create versioned model and log directories
    model_base_dir = ROOT_DIR / "models" / "detr-finetuned"
    model_version_dir = model_base_dir / version
    model_version_dir.mkdir(parents=True, exist_ok=True)

    log_version_dir = LOG_DIR / version
    log_version_dir.mkdir(parents=True, exist_ok=True)

    with open(LABELS_DIR / "classes.yaml") as f:
        classes = yaml.safe_load(f)

    id2label = {i: label for i, label in enumerate(classes)}
    label2id = {label: i for i, label in enumerate(classes)}

    print(f"Training model with {len(classes)} classes. Classes: {classes}")

    checkpoint = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(checkpoint)

    train_dataset = AnimalSpotterDataset(
        img_folder=IMAGES_DIR / "train",
        annotation_file=LABELS_DIR / "train.json",
        processor=processor,
    )

    val_dataset = AnimalSpotterDataset(
        img_folder=IMAGES_DIR / "validation",
        annotation_file=LABELS_DIR / "validation.json",
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

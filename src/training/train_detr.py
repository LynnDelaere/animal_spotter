"""Training script for fine-tuning DETR model on animal spotter dataset."""

from pathlib import Path

# # Add the src directory to sys.path for module imports
# ROOT_DIR = Path(__file__).resolve().parents[2]
# sys.path.append(str(ROOT_DIR))
import torch
import yaml
from dataset_class import AnimalSpotterDataset
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    Trainer,
    TrainingArguments,
)

# Configure paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "processed"

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
        output_dir=str(ROOT_DIR / "models" / "detr-finetuned"),
        per_device_train_batch_size=4,
        num_train_epochs=10,
        fp16=torch.cuda.is_available(),
        save_steps=200,
        logging_steps=50,
        learning_rate=1e-5,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        eval_strategy="steps",
        eval_steps=200,
    )

    # Use a lambda to pass processor to collate_fn
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=lambda batch: collate_fn(batch, processor),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model()
    # Save processor to the same output directory (guarantee a str, not None)
    out_dir = training_args.output_dir or str(ROOT_DIR / "models" / "detr-finetuned")
    processor.save_pretrained(out_dir)
    print(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()

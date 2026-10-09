import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import create_dataloaders
from model import create_model


# --------------------------------------------------
# Settings
# --------------------------------------------------

RANDOM_SEED = 42

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001

METADATA_PATH = "data/processed/metadata_with_splits.csv"

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = "checkpoints/efficientnet_b0_best.pth"


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader, val_loader, _ = create_dataloaders(
    METADATA_PATH,
    batch_size=BATCH_SIZE
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# --------------------------------------------------
# Create EfficientNet-B0
# --------------------------------------------------

model = create_model(
    "efficientnet_b0"
)

model = model.to(device)


# --------------------------------------------------
# Calculate class weights
# --------------------------------------------------

metadata = pd.read_csv(
    METADATA_PATH
)

train_metadata = metadata[
    metadata["split"] == "train"
]

class_counts = (
    train_metadata["label"]
    .value_counts()
    .sort_index()
)

print("\nTraining class counts:")
print(class_counts)


total_samples = len(train_metadata)
num_classes = 2

class_weights = []

for label in range(num_classes):

    count = class_counts[label]

    weight = total_samples / (
        num_classes * count
    )

    class_weights.append(weight)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print("\nClass weights:")
print(class_weights)


# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Train one epoch
# --------------------------------------------------

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Training loop
# --------------------------------------------------

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

best_val_loss = float("inf")


for epoch in range(NUM_EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print("-" * 30)


    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )


    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion,
        device
    )


    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        print("Saved new best model.")


print("\nTraining finished.")

print(
    f"Best EfficientNet-B0 model saved to: "
    f"{CHECKPOINT_PATH}"
)
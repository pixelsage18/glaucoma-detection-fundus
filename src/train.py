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
CHECKPOINT_PATH = "checkpoints/resnet18_best.pth"


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# --------------------------------------------------
# Device
# --------------------------------------------------

# Intel Mac -> CPU
device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader, val_loader, test_loader = create_dataloaders(
    METADATA_PATH,
    batch_size=BATCH_SIZE
)


print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = create_model()

model = model.to(device)


# --------------------------------------------------
# Calculate class weights
# --------------------------------------------------

metadata = pd.read_csv(METADATA_PATH)

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


# Number of training examples
total_samples = len(train_metadata)

# Number of classes
num_classes = 2


# Balanced class weight formula:
#
# total_samples
# -------------------------
# num_classes * class_count
#
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
# Loss function
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
# Training function
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


        # Clear gradients from previous batch
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update model parameters
        optimizer.step()


        # Add batch loss
        running_loss += (
            loss.item() * images.size(0)
        )


        # Predictions
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
# Validation function
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


    # No gradients needed during validation
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
        f"Val Accuracy:   {val_accuracy:.4f}"
    )


    # Save model whenever validation loss improves
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        print(
            "Saved new best model."
        )


print("\nTraining finished.")

print(
    f"Best model saved to: {CHECKPOINT_PATH}"
)

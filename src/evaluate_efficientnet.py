import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from dataset import create_dataloaders
from model import create_model


# --------------------------------------------------
# Settings
# --------------------------------------------------

METADATA_PATH = "data/processed/metadata_with_splits.csv"

CHECKPOINT_PATH = (
    "checkpoints/efficientnet_b0_best.pth"
)

RESULTS_DIR = "results"

BATCH_SIZE = 16


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# Test DataLoader
# --------------------------------------------------

_, _, test_loader = create_dataloaders(
    METADATA_PATH,
    batch_size=BATCH_SIZE
)

print(
    "Test images:",
    len(test_loader.dataset)
)


# --------------------------------------------------
# Load EfficientNet-B0
# --------------------------------------------------

model = create_model(
    "efficientnet_b0"
)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# Prediction storage
# --------------------------------------------------

all_labels = []
all_predictions = []
all_probabilities = []


# --------------------------------------------------
# Run test set
# --------------------------------------------------

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        glaucoma_probabilities = (
            probabilities[:, 1]
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            glaucoma_probabilities
            .cpu()
            .numpy()
        )


# Convert to NumPy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(
    all_predictions
)
all_probabilities = np.array(
    all_probabilities
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

sensitivity = tp / (tp + fn)

specificity = tn / (tn + fp)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nConfusion Matrix:")
print(cm)

print("\nResults:")

print(
    f"Accuracy:    {accuracy:.4f}"
)

print(
    f"Sensitivity: {sensitivity:.4f}"
)

print(
    f"Specificity: {specificity:.4f}"
)

print(
    f"ROC-AUC:     {roc_auc:.4f}"
)


print("\nDetailed counts:")

print(
    "True Negatives: ",
    tn
)

print(
    "False Positives:",
    fp
)

print(
    "False Negatives:",
    fn
)

print(
    "True Positives: ",
    tp
)


# --------------------------------------------------
# Create results directory
# --------------------------------------------------

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Confusion matrix figure
# --------------------------------------------------

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_title(
    "EfficientNet-B0 Confusion Matrix"
)

ax.set_xlabel(
    "Predicted label"
)

ax.set_ylabel(
    "True label"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "Healthy",
    "Glaucoma"
])

ax.set_yticklabels([
    "Healthy",
    "Glaucoma"
])


for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix_efficientnet.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=200
)

plt.close()


# --------------------------------------------------
# ROC curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probabilities
)


plt.figure()

plt.plot(
    fpr,
    tpr,
    label=(
        f"EfficientNet-B0 "
        f"AUC = {roc_auc:.3f}"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "EfficientNet-B0 ROC Curve"
)

plt.legend()

plt.tight_layout()


roc_curve_path = os.path.join(
    RESULTS_DIR,
    "roc_curve_efficientnet.png"
)

plt.savefig(
    roc_curve_path,
    dpi=200
)

plt.close()


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics_efficientnet.md"
)


with open(metrics_path, "w") as file:

    file.write(
        "# EfficientNet-B0 Test Results\n\n"
    )

    file.write(
        f"- Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"- Sensitivity: {sensitivity:.4f}\n"
    )

    file.write(
        f"- Specificity: {specificity:.4f}\n"
    )

    file.write(
        f"- ROC-AUC: {roc_auc:.4f}\n"
    )

    file.write(
        "\n## Confusion Matrix\n\n"
    )

    file.write(
        f"- True negatives: {tn}\n"
    )

    file.write(
        f"- False positives: {fp}\n"
    )

    file.write(
        f"- False negatives: {fn}\n"
    )

    file.write(
        f"- True positives: {tp}\n"
    )


print("\nSaved results:")

print(confusion_matrix_path)
print(roc_curve_path)
print(metrics_path)

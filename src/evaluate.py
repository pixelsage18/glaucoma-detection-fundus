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
CHECKPOINT_PATH = "checkpoints/resnet18_best.pth"

RESULTS_DIR = "results"

BATCH_SIZE = 16


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# Load test DataLoader
# --------------------------------------------------

_, _, test_loader = create_dataloaders(
    METADATA_PATH,
    batch_size=BATCH_SIZE
)


print("Test images:", len(test_loader.dataset))


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = create_model()

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# Store predictions
# --------------------------------------------------

all_labels = []
all_predictions = []
all_probabilities = []


# --------------------------------------------------
# Run model on test set
# --------------------------------------------------

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Convert raw model outputs into probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Probability of glaucoma class (class 1)
        glaucoma_probabilities = probabilities[:, 1]

        # Predicted class
        predictions = outputs.argmax(dim=1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            glaucoma_probabilities.cpu().numpy()
        )


# Convert lists to NumPy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)


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

print(f"Accuracy:    {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"ROC-AUC:     {roc_auc:.4f}")

print("\nDetailed counts:")
print("True Negatives: ", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives: ", tp)


# --------------------------------------------------
# Create results directory
# --------------------------------------------------

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Save confusion matrix figure
# --------------------------------------------------

fig, ax = plt.subplots()

image = ax.imshow(cm)

ax.set_title("ResNet18 Confusion Matrix")

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

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


# Put numbers inside cells
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
    "confusion_matrix.png"
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
    label=f"ResNet18 AUC = {roc_auc:.3f}"
)

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()


roc_curve_path = os.path.join(
    RESULTS_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_curve_path,
    dpi=200
)

plt.close()


# --------------------------------------------------
# Save metrics to Markdown
# --------------------------------------------------

metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics.md"
)


with open(metrics_path, "w") as file:

    file.write("# ResNet18 Test Results\n\n")

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

    file.write("\n## Confusion Matrix\n\n")

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

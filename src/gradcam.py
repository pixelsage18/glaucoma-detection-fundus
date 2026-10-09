import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import eval_transform
from model import create_model


# --------------------------------------------------
# Settings
# --------------------------------------------------

METADATA_PATH = "data/processed/metadata_with_splits.csv"
CHECKPOINT_PATH = "checkpoints/resnet18_best.pth"

OUTPUT_PATH = "results/gradcam_examples.png"

device = torch.device("cpu")


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
# Load test glaucoma images
# --------------------------------------------------

df = pd.read_csv(METADATA_PATH)

glaucoma_test_df = df[
    (df["split"] == "test") &
    (df["label"] == 1)
].reset_index(drop=True)

print("Glaucoma images in test set:", len(glaucoma_test_df))


# --------------------------------------------------
# Find correctly predicted glaucoma images
# --------------------------------------------------

selected_examples = []

for _, row in glaucoma_test_df.iterrows():

    image = Image.open(
        row["image_path"]
    ).convert("RGB")

    input_tensor = (
        eval_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = outputs.argmax(dim=1).item()
        glaucoma_probability = probabilities[0, 1].item()

    # Keep correctly predicted glaucoma examples
    if predicted_class == 1:
        selected_examples.append(
            {
                "row": row,
                "image": image,
                "input_tensor": input_tensor,
                "glaucoma_probability": glaucoma_probability
            }
        )

    # We only need 4 examples
    if len(selected_examples) == 4:
        break


if len(selected_examples) == 0:
    raise RuntimeError(
        "No correctly predicted glaucoma images were found."
    )

print("Correctly predicted glaucoma examples selected:", len(selected_examples))


# --------------------------------------------------
# Prepare Grad-CAM
# --------------------------------------------------

target_layers = [model.layer4[-1]]
targets = [ClassifierOutputTarget(1)]

os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# Create figure
# --------------------------------------------------

num_examples = len(selected_examples)

fig, axes = plt.subplots(
    num_examples,
    2,
    figsize=(10, 4 * num_examples)
)

# If only 1 example is found, axes won't be a 2D array
if num_examples == 1:
    axes = np.array([axes])


with GradCAM(
    model=model,
    target_layers=target_layers
) as cam:

    for i, example in enumerate(selected_examples):

        image = example["image"]
        input_tensor = example["input_tensor"]
        glaucoma_probability = example["glaucoma_probability"]
        image_path = example["row"]["image_path"]

        # Resize image for display
        display_image = image.resize((224, 224))
        display_image = np.array(display_image).astype(np.float32) / 255.0

        # Generate Grad-CAM
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        cam_image = show_cam_on_image(
            display_image,
            grayscale_cam,
            use_rgb=True
        )

        # Original image
        axes[i, 0].imshow(display_image)
        axes[i, 0].set_title(
            f"Original\n{os.path.basename(image_path)}"
        )
        axes[i, 0].axis("off")

        # Grad-CAM image
        axes[i, 1].imshow(cam_image)
        axes[i, 1].set_title(
            f"Grad-CAM\nGlaucoma prob: {glaucoma_probability:.3f}"
        )
        axes[i, 1].axis("off")


plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print(f"Saved Grad-CAM examples to: {OUTPUT_PATH}")

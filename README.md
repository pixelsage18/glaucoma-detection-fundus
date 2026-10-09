# Glaucoma Detection from Retinal Fundus Images

A deep learning project for binary classification of retinal fundus images into **Healthy** and **Glaucoma** using transfer learning in PyTorch.

Two pretrained CNN models were compared:

- ResNet18
- EfficientNet-B0

The project also includes patient-level splitting, class imbalance handling, medical evaluation metrics, and Grad-CAM visualizations.

---

## Dataset

The project uses the **PAPILA** retinal fundus dataset.

For this binary classification task:

- Healthy = 0
- Glaucoma = 1
- Suspect cases were excluded

After removing suspect cases and records without matching fundus images, the final dataset contained:

| Class | Images |
|---|---:|
| Healthy | 306 |
| Glaucoma | 87 |
| Total | 393 |

The data was split by **patient ID**, so images from the same patient could not appear in both training and test sets.

| Split | Patients | Images |
|---|---:|---:|
| Train | 137 | 273 |
| Validation | 30 | 60 |
| Test | 30 | 60 |

---

## Method

Images were resized to `224 × 224` and normalized using ImageNet normalization.

Training used:

- Transfer learning
- Adam optimizer
- Learning rate: `0.0001`
- Batch size: `16`
- 5 epochs
- Weighted Cross-Entropy Loss
- Random horizontal flipping for training augmentation

Class weighting was used because glaucoma images were much less common than healthy images.

---

## Results

| Model | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|
| ResNet18 | 0.7000 | 0.8462 | 0.6596 | 0.8331 |
| EfficientNet-B0 | 0.7000 | 0.9231 | 0.6383 | 0.8265 |

EfficientNet-B0 achieved higher sensitivity, while ResNet18 achieved slightly better specificity and ROC-AUC.

Because the test set contains only 60 images, these results should be interpreted cautiously.

---

## Grad-CAM

Grad-CAM was applied to the trained ResNet18 model to visualize regions influencing glaucoma predictions.

Several examples showed attention around the **optic disc / optic nerve head region**, although some activation was also spread across other retinal areas.

Example visualizations are stored in:

```text
results/gradcam_examples.png
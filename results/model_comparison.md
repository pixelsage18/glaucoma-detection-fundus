# Model Comparison

Both models were evaluated on the same patient-level test split containing 60 fundus images.

| Model | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|
| ResNet18 | 0.7000 | 0.8462 | 0.6596 | 0.8331 |
| EfficientNet-B0 | 0.7000 | 0.9231 | 0.6383 | 0.8265 |

## ResNet18

Confusion matrix:

- True Negatives: 31
- False Positives: 16
- False Negatives: 2
- True Positives: 11

ResNet18 achieved slightly higher specificity and ROC-AUC.

## EfficientNet-B0

Confusion matrix:

- True Negatives: 30
- False Positives: 17
- False Negatives: 1
- True Positives: 12

EfficientNet-B0 achieved higher sensitivity, correctly detecting 12 of the 13 glaucoma images in the test set.

## Interpretation

Both models achieved the same overall accuracy.

EfficientNet-B0 showed higher sensitivity but slightly lower specificity. ResNet18 achieved a slightly higher ROC-AUC and was retained as the primary model for the Grad-CAM analysis.

Because the test set contains only 60 images, these results should be interpreted cautiously. Small changes in the number of correct predictions can noticeably affect the reported metrics.
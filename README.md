# Glaucoma Detection from Retinal Fundus Images

A deep learning project for binary classification of retinal fundus images into **Healthy** and **Glaucoma** using transfer learning in PyTorch.

Two pretrained CNNs were compared:

- ResNet18
- EfficientNet-B0

The project includes patient-level data splitting, class imbalance handling, medical evaluation metrics (sensitivity, specificity, ROC-AUC) and Grad-CAM visualizations.

> **Disclaimer:** This is a student learning project. It is not a medical device and must not be used for diagnosis or clinical decisions.

---

## Table of contents

1. [Dataset](#dataset)
2. [Method](#method)
3. [Results](#results)
4. [Grad-CAM](#grad-cam)
5. [Limitations](#limitations)
6. [Future work](#future-work)
7. [Project structure](#project-structure)
8. [How to run](#how-to-run)
9. [Acknowledgements](#acknowledgements)

---

## Dataset

The project uses the **PAPILA** retinal fundus dataset, which provides fundus images together with clinical data for both eyes of each patient. The raw data is **not** included in this repository.

For this binary task:

- Healthy = 0
- Glaucoma = 1
- Suspect cases were excluded

After removing suspect cases and records without a matching fundus image, the final dataset contained:

| Class | Images |
|---|---:|
| Healthy | 306 |
| Glaucoma | 87 |
| Total | 393 |

### Patient-level split

Each patient can contribute images of both eyes. To avoid data leakage, the data was split by **patient ID**, so images from the same patient never appear in more than one split. Patients were grouped as healthy-only, glaucoma-only or mixed (one healthy and one glaucoma eye), and the split was stratified on these groups. The code checks that no patient appears in multiple splits.

| Split | Patients | Images |
|---|---:|---:|
| Train | 137 | 273 |
| Validation | 30 | 60 |
| Test | 30 | 60 |

The test set contains **13 glaucoma and 47 healthy images**.

---

## Method

- Images resized to `224 x 224` and normalized with ImageNet statistics
- Transfer learning: ImageNet-pretrained ResNet18 and EfficientNet-B0 with the final layer replaced by a 2-class layer
- Optimizer: Adam, learning rate `0.0001`
- Batch size: `16`
- Epochs: `5`
- Loss: weighted cross-entropy, with class weights computed from the training set because glaucoma images are much less common than healthy ones
- Augmentation: random horizontal flip (training only)
- The checkpoint with the lowest validation loss was kept, and the test set was used only for the final evaluation
- Fixed random seed (`42`) for reproducibility
- Trained on CPU

---

## Results

Test set results (60 images, patient-level split):

| Model | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|
| ResNet18 | 0.7000 | 0.8462 | 0.6596 | 0.8331 |
| EfficientNet-B0 | 0.7000 | 0.9231 | 0.6383 | 0.8265 |

Confusion matrices:

| Model | True negatives | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|
| ResNet18 | 31 | 16 | 2 | 11 |
| EfficientNet-B0 | 30 | 17 | 1 | 12 |

### How to read these results

- **Accuracy is not the main metric here.** The test set is about 78% healthy, so a model that predicted "healthy" for every image would score about 78% accuracy. Sensitivity, specificity and ROC-AUC give a more useful picture.
- **The two models perform about the same.** The sensitivity difference (11 vs. 12 of 13 glaucoma images) is a single image, and the ROC-AUC difference (0.833 vs. 0.827) is too small to be meaningful on 60 images.
- **Specificity is around 65%.** About one in three healthy eyes was flagged as glaucoma. For a screening setting this would mean many false alarms.
- ResNet18 was used for the Grad-CAM analysis.

Full outputs are in the `results/` folder (metrics, confusion matrices, ROC curves).

---

## Grad-CAM

Grad-CAM was applied to the trained ResNet18 to see which regions influenced glaucoma predictions. The examples shown are the **first four correctly predicted glaucoma images in the test set**, not hand-picked ones.

In three of the four examples, the strongest activation was centred on the **optic disc / optic nerve head region**, which is where clinical signs of glaucoma appear. In the fourth (`RET019OD`, glaucoma probability 0.54), the strongest activation was below the disc, so the model was not clearly focusing on the optic disc in that case.

Grad-CAM shows where the model looked, not that the model reasons like a clinician, so it should be treated as a sanity check only.

Examples are saved in `results/gradcam_examples.png`.

---

## Limitations

- **Small dataset:** 393 images in total and only 60 test images (13 with glaucoma). Metrics change noticeably with a single prediction, and no confidence intervals are reported.
- **Single dataset, no external validation:** the models were not tested on images from other cameras, hospitals or populations, so performance elsewhere is unknown.
- **Population:** PAPILA was not collected from an Indian population, so these results say nothing about performance on Indian patients.
- **Suspect cases excluded:** borderline cases, which are the hardest in practice, were removed.
- **Limited training:** only 5 epochs, a single augmentation (horizontal flip), no hyperparameter search and no cross-validation.
- **High false-positive rate:** specificity is around 65%.
- **Model choice after viewing test results:** the final model comparison was made on the same 60-image test set, and the differences between models are within noise.
- **Images only:** the models do not use clinical measurements such as eye pressure.

---

## Future work

- **Multimodal model:** combine image features with the clinical data that comes with PAPILA (for example eye pressure measurements, if available) and test whether it improves results.
- **External validation** on other public fundus datasets, and on Indian-population data if it becomes available.
- **Cross-validation** and **bootstrap confidence intervals** for more reliable estimates on a small dataset.
- **Optic disc and cup segmentation**, to compute the cup-to-disc ratio as an interpretable feature.
- Longer training, stronger augmentation, and threshold tuning to trade sensitivity against specificity.
- Training and validation curves to check for under- and overfitting.

---

## Project structure

```text
glaucoma-detection-fundus/
├── README.md
├── requirements.txt
├── .gitignore
├── notes/
│   └── papers.md            # background reading notes
├── data/
│   └── processed/
│       ├── metadata.csv
│       └── metadata_with_splits.csv
├── src/
│   ├── build_metadata.py        # builds image-label table from PAPILA clinical files
│   ├── create_splits.py         # patient-level train/val/test split
│   ├── dataset.py               # dataset class, transforms, dataloaders
│   ├── model.py                 # ResNet18 and EfficientNet-B0 with new classifier head
│   ├── train.py                 # ResNet18 training
│   ├── evaluate.py              # ResNet18 evaluation (metrics, confusion matrix, ROC)
│   ├── train_efficientnet.py    # EfficientNet-B0 training
│   ├── evaluate_efficientnet.py # EfficientNet-B0 evaluation
│   └── gradcam.py               # Grad-CAM visualizations (ResNet18)
└── results/
    ├── metrics.md, metrics_efficientnet.md, model_comparison.md
    ├── confusion_matrix*.png, roc_curve*.png
    └── gradcam_examples.png
```

The raw dataset (`data/raw/`) and model checkpoints (`checkpoints/`) are excluded from the repository.

---

## How to run

Run all commands from the repository root. The project was developed with Python 3.12 on CPU.

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Get the data.** Download the PAPILA dataset from its official source (see [Acknowledgements](#acknowledgements)) and place it so that the following folders exist:

   ```text
   data/raw/PAPILA/ClinicalData/    # patient_data_od.xlsx, patient_data_os.xlsx
   data/raw/PAPILA/FundusImages/    # RET###OD.jpg, RET###OS.jpg
   ```

3. **Prepare the metadata and splits**

   ```bash
   python src/build_metadata.py
   python src/create_splits.py
   ```

4. **Train and evaluate ResNet18**

   ```bash
   python src/train.py
   python src/evaluate.py
   python src/gradcam.py
   ```

5. **Train and evaluate EfficientNet-B0**

   ```bash
   python src/train_efficientnet.py
   python src/evaluate_efficientnet.py
   ```

Results are written to the `results/` folder.

---

## Acknowledgements

- **PAPILA dataset:** Kovalyk et al., *PAPILA: Dataset with fundus images and clinical data of both eyes of the same patient for glaucoma assessment*, Scientific Data, 2022. Please check the dataset's official page for its license and terms of use before reusing the data.
- Pretrained model weights from `torchvision` (ImageNet).
- Grad-CAM implementation from the `pytorch-grad-cam` package.

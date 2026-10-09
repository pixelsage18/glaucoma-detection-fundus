# Background notes: glaucoma detection from fundus images

Project context: early glaucoma detection using deep learning on fundus images, with future work on multimodal data (images plus clinical measurements).

Note on how these notes were made: the paper summaries below were drafted from the REFUGE paper's text and the REFUGE2 abstract and summaries. Re-read the abstracts yourself before the interview and edit anything that does not match what you understand.

## 1. Glaucoma basics (in my own words)

- Glaucoma is a chronic condition that damages the optic nerve head, usually linked to raised eye pressure (IOP). The damage is irreversible, but early detection can prevent vision loss. It often has no symptoms until late, which is why it is called the "silent thief of sight". At least half of patients remain undiagnosed.
- A **fundus photograph** (color fundus photography, CFP) is a picture of the back of the eye. It shows the **optic disc** (where the optic nerve leaves the eye) and the **optic cup** (the paler depression in its middle).
- The **vertical cup-to-disc ratio (vCDR)** compares the cup diameter to the disc diameter. A larger vCDR is a classic sign of glaucomatous damage. Other signs include thinning of the retinal nerve fibre layer (RNFL) and small bleeds near the disc.
- In clinics, fundus photos are used together with IOP measurements, visual field tests and OCT scans. Doing all of these for population screening is expensive, so automated tools that get more out of the fundus photo alone are valuable.
- **Multimodal** in this project most likely means combining several sources (fundus images, OCT, clinical features such as IOP) instead of using images alone.

## 2. Paper notes

### Paper 1: REFUGE challenge

Orlando et al., "REFUGE Challenge: A Unified Framework for Evaluating Automated Methods for Glaucoma Assessment from Fundus Photographs", Medical Image Analysis, 2020. arXiv: 1910.03667.

- **Problem:** Deep learning was little used for glaucoma because public datasets were small, and there was no standard benchmark, so methods could not be compared fairly.
- **Data:** 1200 fundus images (120 glaucoma, 1080 non-glaucoma, so about 10% positive), split into three fixed sets of 400 (training, offline test, online test). The training images came from one camera (Zeiss) and the test images from another (Canon), on purpose, to test generalization across devices. Labels were assigned from clinical records (follow-up images, IOP, OCT, visual fields), not from one photo. The patients were Chinese patients from clinics.
- **Method and result:** A challenge with two tasks, glaucoma classification and optic disc/cup segmentation. 12 teams took part in the on-site final. Most classification teams fine-tuned CNNs pretrained on ImageNet (ResNet, DenseNet, Xception and others), often as ensembles, because the training set had only 400 images. The best team reached an AUC of about 0.99, and the second about 0.98. Using the ground-truth vCDR alone gave an AUC of about 0.95. Two top teams outperformed two human experts, who graded images only and reached 85% sensitivity at about 91% specificity.
- **Limitation:** All images came from Chinese patients and only two camera types, so results may not carry over to other populations or devices. The glaucoma share (10%) is much higher than real-world prevalence (about 4%), and only high-quality images were used.
- **Why it matters for my project:** It confirms that fine-tuning pretrained CNNs is the standard approach for small fundus datasets, and that AUC, sensitivity and specificity are the right metrics. It also shows why testing on a different device matters, and why an Indian-population dataset is a gap.

### Paper 2: REFUGE2

Orlando et al. were organizers of the first edition. REFUGE2 is by Fang et al., arXiv: 2202.08994. Note: the first arXiv version is titled "REFUGE2 Challenge: Treasure for Multi-Domain Learning in Glaucoma Assessment", and a later version uses the title "A Treasure Trove for Multi-Dimension Analysis and Evaluation in Glaucoma Screening". Cite whichever version you actually read.

- **Problem:** Existing datasets used different labeling standards, often did not say which device took the images, and datasets with detailed annotations were small.
- **Data:** 2000 color fundus images from four camera models (Zeiss, Canon, Kowa, Topcon), with glaucoma labels, optic disc/cup segmentations and fovea locations. Three sub-tasks: glaucoma classification, disc/cup segmentation and fovea localization.
- **Method and result:** Over 1,300 participants joined, 134 teams submitted preliminary results and 22 teams reached the final. Teams that used domain adaptation (methods that help a model handle images from different devices) performed well and stayed robust across devices.
- **Limitation:** To check in the paper: whether the population is still mostly one country, and how results held up on the hardest device. Write your own line after reading.
- **Why it matters for my project:** It has a table comparing public fundus datasets, which I will use to choose my dataset. It also shows that a model trained on one device can fail on another, which I should mention in my README limitations.

### Paper 3: Christopher et al. 2018 (as described in the REFUGE paper)

"Performance of Deep Learning Architectures and Transfer Learning for Detecting Glaucomatous Optic Neuropathy in Fundus Photographs" (Christopher et al., 2018). I have only seen it described in the REFUGE paper, so look up the original abstract and check the journal and details before citing it.

- **Problem:** Detect glaucomatous optic neuropathy from fundus photos using deep learning.
- **Data:** According to REFUGE, a large private dataset of more than 14,000 images.
- **Method and result:** Fine-tuned a network initialized with ImageNet weights, which is the same transfer learning approach I will use.
- **Limitation:** According to REFUGE, labels in this study were graded from how the images looked, without clinical records, which can teach a model the same labeling mistakes. The data was also not public, so results cannot be reproduced.

## 3. Datasets seen in the REFUGE paper's comparison table (to check before choosing)

Image counts from REFUGE's Table 1 (check current availability and licenses on each dataset's page):

- REFUGE: 1200 images (120 glaucoma), with clinical labels and a fixed train/test split.
- RIM-ONE (v1, v2, v3): 158, 455 and 169 images, with glaucoma labels. v3 contains stereo images.
- DRISHTI-GS: 101 images, with a train/test split, but labels based only on the images.
- ORIGA: 650 images, but the paper says it is no longer publicly available.
- REFUGE2: 2000 images from four camera models (from the REFUGE2 abstract).

## 4. What this means for my project

- I will treat this as **binary classification** (glaucoma vs. normal) on fundus images, with transfer learning from ImageNet weights, as the strongest teams did.
- Accuracy alone is misleading because glaucoma images are a small share of the data. I will report **sensitivity, specificity and ROC-AUC**, plus a confusion matrix, and use a weighted loss for the imbalance.
- I will use **Grad-CAM** to check whether the model looks at the optic disc region, since that is where the clinical signs are.
- I will split data so the same patient never appears in both training and test sets, and say honestly if the dataset does not let me do that.
- The REFUGE data is from one population, and the project is about the Indian population. I will state this as a limitation, and as the reason future work needs Indian-population data.
- Future work: combine image features with clinical data (such as IOP) in a multimodal model.

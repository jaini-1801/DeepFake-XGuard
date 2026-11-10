# DeepFake XGuard 

**An EfficientNet-B0-based deepfake detection system with cross-dataset evaluation and robustness analysis.**

## 📌 Overview

DeepFake XGuard is a deepfake detection project that uses deep learning and computer vision to classify facial images as **Real** or **Fake**.

The system uses EfficientNet-B0 as its backbone and is trained on the FaceForensics++ dataset. Its generalisation ability is further evaluated using Celeb-DF v2.

Beyond classification, the project investigates how image degradation affects detection performance through JPEG compression, image resizing and Gaussian blur. It also includes supporting visual analyses using Error Level Analysis (ELA) and Principal Component Analysis (PCA).

##  Objectives

- Develop a deep learning-based real/fake face classification system.
- Train and evaluate EfficientNet-B0 on FaceForensics++.
- Test cross-dataset generalisation using Celeb-DF v2.
- Analyse robustness under different image degradation conditions.
- Explore image-level and feature-level patterns using ELA and PCA.

##  Datasets

### 1. FaceForensics++

Used for model training, validation and in-dataset testing.

The dataset includes:
- Original
- Deepfakes
- Face2Face
- FaceShifter
- FaceSwap
- NeuralTextures

### 2. Celeb-DF v2

Used for external evaluation to examine how well the trained model performs on an unseen dataset.

**Note:** The model is evaluated on Celeb-DF v2 without additional fine-tuning.

Dataset access:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)

Please follow the respective dataset licences and access requirements.

##  Methodology

The project follows this pipeline:

1. **Dataset preparation:** Organising video data and creating dataset manifests.
2. **Frame extraction:** Sampling frames from source videos.
3. **Face extraction:** Using MTCNN to detect and crop facial regions.
4. **Preprocessing:** Resizing face images to 160 × 160 pixels and normalising inputs.
5. **Model training:** Fine-tuning EfficientNet-B0 for binary classification.
6. **Evaluation:** Measuring classification performance using standard metrics.
7. **Robustness testing:** Evaluating performance under JPEG compression, resizing and Gaussian blur.
8. **External evaluation:** Testing on Celeb-DF v2.
9. **Additional analysis:** Exploring ELA and PCA visualisations.

##  Model Architecture

**EfficientNet-B0**

- Pretrained ImageNet weights
- Modified final classification layer
- Two output classes: Real (0) and Fake (1)
- Input resolution: 160 × 160 pixels
- Optimiser: AdamW
- Learning rate: 0.0001
- Batch size: 8
- Training epochs: 10

A class-weighted training version was also explored to account for class imbalance.

##  Results

### FaceForensics++ evaluation

| Metric | Score |
|---|---:|
| Accuracy | 88.20% |
| Precision | 99.97% |
| Recall | 85.87% |
| F1-score | 92.38% |
| ROC-AUC | 99.52% |
| Average Precision | 99.85% |
| Balanced Accuracy | 92.86% |

### Celeb-DF v2 external evaluation

| Metric | Score |
|---|---:|
| Accuracy | 70.27% |
| Precision | 77.68% |
| Recall | 76.76% |
| F1-score | 77.22% |
| ROC-AUC | 74.44% |
| Balanced Accuracy | 67.31% |

The difference between FaceForensics++ and Celeb-DF v2 performance highlights the challenge of cross-dataset generalisation.

##  Robustness Analysis

The model was evaluated under three image degradation conditions:

- **JPEG compression:** Tests sensitivity to compression quality.
- **Image resizing:** Tests the effect of reduced image resolution.
- **Gaussian blur:** Tests sensitivity to loss of image details.

The experiments indicate that severe resizing and Gaussian blur reduce model performance, with Gaussian blur having the strongest impact.

##  Additional Analysis

### Error Level Analysis (ELA)
Used to visualise image compression differences and explore possible manipulation-related patterns.

### Principal Component Analysis (PCA)
Used to project EfficientNet feature embeddings into two dimensions to explore feature distribution and class overlap.

These analyses are exploratory and are not used as independent proof of deepfake manipulation.

##  Technologies Used

- Python
- PyTorch
- EfficientNet-B0
- OpenCV
- MTCNN
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

##  Project Structure

```text
DeepFake-XGuard/
│
├── Data/
│   └── manifests/
│
├── src/
│   ├── create_manifests.py
│   ├── extract_frames.py
│   ├── extract_faces.py
│   ├── train.py
│   ├── train_weighted.py
│   ├── train_evaluate.py
│   ├── evaluate.py
│   ├── evaluate_celebdf.py
│   ├── robustness_test.py
│   └── ela_analysis.py
│
├── figures/
│   ├── pca_efficientnet_features.png
│   ├── ela_distribution.png
│   └── other analysis figures
│
├── results/
│   ├── robustness_results.csv
│   └── celebdf/
│
├── README.md
├── requirements.txt
└── .gitignore
```

*The structure above is representative. Keep the actual filenames and folders present in your repository.*

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DeepFake-XGuard.git
cd DeepFake-XGuard
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the datasets

Download FaceForensics++ and Celeb-DF v2 separately, following their access requirements.

Organise the datasets and update the paths in the relevant Python scripts before running the pipeline.

### 5. Run the pipeline

Run the relevant scripts in the following order, adjusting arguments and paths according to their implementation:

```bash
python src/create_manifests.py
python src/extract_frames.py
python src/extract_faces.py
python src/train_weighted.py
python src/train_evaluate.py
python src/evaluate_celebdf.py
python src/robustness_test.py
python src/ela_analysis.py
```

**Important:** The commands above are a suggested pipeline order. Check each script for its required arguments, paths and checkpoint settings before running.

##  Limitations

- Performance decreases when evaluated on a different dataset.
- Severe blur and resizing affect detection performance.
- ELA patterns are influenced by compression and image processing.
- PCA visualisations provide only a two-dimensional view of high-dimensional features.
- Further evaluation on diverse datasets and real-world videos is required.

##  Future Work

- Improve cross-dataset generalisation.
- Evaluate on additional deepfake datasets.
- Explore temporal information from video sequences.
- Investigate more robust feature representations.
- Improve performance under real-world image degradation.

##  Disclaimer

This project is developed for academic and research purposes. Its predictions should not be treated as definitive proof that an image or video is authentic or manipulated.
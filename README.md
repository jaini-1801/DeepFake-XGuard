# DeepFake XGuard

**EfficientNet-B0-based deepfake detection with cross-dataset evaluation, robustness analysis, and forensic visualisation.**

---

## Overview

DeepFake XGuard detects synthetic face manipulations using a fine-tuned EfficientNet-B0 backbone. The model is trained on FaceForensics++ and evaluated both in-domain and zero-shot on Celeb-DF v2 to measure real-world generalisation. Robustness under JPEG compression, resizing, and Gaussian blur is characterised, and forensic views (ELA mean maps and PCA of deep features) are provided for interpretability.

---

## Objectives

- Train and evaluate EfficientNet-B0 on FaceForensics++ for binary real/fake classification.
- Test zero-shot cross-dataset generalisation on Celeb-DF v2.
- Characterise robustness under common image degradations.
- Provide interpretability via Error Level Analysis (ELA) and PCA of learned features.

---

## Datasets

| Dataset | Role | Size |
|---|---|---|
| FaceForensics++ | Train / Val / Test | 10,263 face crops (original split); 13,261 in mixed training |
| Celeb-DF v2 | External zero-shot test | 518 videos (340 fake, 178 real) |

**Sources:**
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)

Please follow each dataset's licence and access requirements.

---

## Methodology

### Data Pipeline

```
FF++ Videos ──► Frame Extraction (1 fps) ──► MTCNN Face Detection ──► Crop & Resize 160×160
                                                                             │
Celeb-DF v2 ──► Frame Extraction (1 fps) ──► MTCNN Face Detection ──► Crop & Resize 160×160
                                                                             │
                                                                    Augment + Normalise
                                                                             │
                                                                   EfficientNet-B0 Fine-tuning
                                                                     /        |         \
                                                               Eval FF++  Eval Celeb-DF  Robustness Test
```

### Model Architecture

EfficientNet-B0 uses compound scaling (width × depth × resolution) across 7 MBConv blocks with Squeeze-and-Excitation attention. The 1280-dim feature vector from the global average pool feeds a binary classifier.

| Component | Detail |
|---|---|
| Backbone | EfficientNet-B0, ImageNet pre-trained |
| Input | 160 × 160 RGB face crops |
| Blocks | 7 × MBConv (ch 16 → 320) with SE attention and identity skip connections |
| Head | AvgPool → FC(1280) → FC(2) |
| Parameters | 5.3M |

### Training Configuration

| Hyper-parameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Loss | Cross-Entropy with class weights (fake 0.774 / real 1.412) |
| Epochs | 10 |
| Batch size | 8 |
| Input size | 160 × 160 |
| Augmentation | Random horizontal flip, colour jitter, Gaussian blur, random erasing |

---

## Results

### FaceForensics++ (in-domain)

| Metric | threshold = 0.50 | threshold = 0.84 |
|---|---:|---:|
| Accuracy | 87.75% | 71.58% |
| Precision | 88.53% | **97.88%** |
| Recall | **98.00%** | 67.36% |
| F1-score | 93.02% | 79.80% |
| Balanced Accuracy | 67.25% | **80.04%** |
| **ROC-AUC** | **90.07%** | 90.07% |

Confusion matrix at thr = 0.50 (n = 8,400):

|  | Pred Real | Pred Fake |
|---|---:|---:|
| **True Real** | TN = 511 | FP = 889 |
| **True Fake** | FN = 140 | TP = 6,860 |

> **Best operating point:** thr = 0.84 — precision 97.88%, balanced accuracy 80.04%, FP drops from 889 to 102.

---

### Celeb-DF v2 (zero-shot cross-dataset)

| Metric | FF++ | Celeb-DF v2 | Change |
|---|---:|---:|---:|
| Accuracy | 87.75% | 70.27% | ↓ 17.48 pp |
| Precision | 88.53% | 77.68% | ↓ 10.85 pp |
| Recall | 98.00% | 76.76% | ↓ 21.24 pp |
| F1-score | 93.02% | 77.22% | ↓ 15.80 pp |
| **ROC-AUC** | **90.07%** | **74.44%** | **↓ 15.63 pp** |
| Balanced Accuracy | 67.25% | 67.31% | ≈ 0 |

> **Root cause:** Domain shift — FF++ uses raw H.264 video; Celeb-DF uses YouTube-compressed clips with different facial styles and GAN generators.

---

### Advanced Metrics (FF++ vs Celeb-DF v2)

| Metric | FF++ | Celeb-DF v2 |
|---|---:|---:|
| Equal Error Rate (EER) | 2.79% | 31.18% |
| pAUC @ FPR ≤ 0.1 | 0.9661 | 0.1801 |
| pAUC @ FPR ≤ 0.2 | — | — |
| Temperature (calibration) | — | T = 2.06 (overconfident) |

---

### Robustness Under Image Degradation

| Degradation | Setting | F1 (%) | ROC-AUC (%) |
|---|---|---:|---:|
| None (clean) | — | 93.02 | 90.07 |
| JPEG compression | quality = 90 | 92.66 | 89.65 |
| JPEG compression | quality = 70 | 92.84 | 88.40 |
| JPEG compression | quality = 50 | 92.48 | 86.91 |
| JPEG compression | quality = 30 | 91.68 | 83.39 |
| Resizing | scale = 0.75 | 92.28 | 86.75 |
| Resizing | scale = 0.50 | 91.23 | 79.52 |
| Resizing | scale = 0.25 | 88.51 | 66.57 |
| Gaussian blur | σ = 1 | 91.60 | 83.78 |
| Gaussian blur | σ = 2 | 90.78 | 71.03 |
| Gaussian blur | σ = 3 | 88.07 | 59.84 |
| Gaussian blur | σ = 4 | 83.87 | **51.50** |

> Gaussian blur (σ = 4) is the most damaging — AUC collapses from 90.07% to 51.50% (↓ 38.57 pp). This confirms the model relies on high-frequency compression artefacts that blur destroys.

---

## Forensic Analysis

### Error Level Analysis (ELA)

ELA reveals inconsistent JPEG re-compression across an image. Manipulated (spliced) regions carry a different compression history and appear brighter in ELA maps. The **ELA mean maps** averaged over each class show that fake faces exhibit systematically elevated residuals at forehead, jaw-line, and blending boundaries — validating that the model responds to real forensic artefacts.

### PCA of EfficientNet-B0 Features

The 1280-dimensional penultimate-layer embeddings are projected to 2D with PCA. Visible cluster separation between real and fake faces confirms the backbone learnt a discriminative representation. Cluster overlap identifies the hard examples that cause false positives/negatives.

---

## Project Structure

```
DeepFake-XGuard/
├── Src/
│   ├── create_manifests.py       # Dataset manifest generation
│   ├── extract_frames.py         # Frame sampling from videos
│   ├── extract_faces.py          # MTCNN face cropping (FF++)
│   ├── extracted_celebdf_frames.py
│   ├── extracted_celebdf_faces.py
│   ├── train.py                  # Baseline training
│   ├── train_weighted.py         # Class-weighted training (AdamW + CE)
│   ├── train_evaluate.py         # Training with evaluation loop
│   ├── train_robust.py           # Degradation-aware training
│   ├── evaluate.py               # In-domain evaluation (FF++)
│   ├── evaluate_celebdf.py       # Cross-dataset evaluation (Celeb-DF)
│   ├── robustness_test.py        # Robustness sweep
│   ├── robustness_test_robust.py
│   ├── Validate_robust_clean.py
│   ├── roc_analysis.py           # ROC / EER curves
│   ├── ffpp_roc_analysis.py
│   ├── plot_robustness.py        # Robustness visualisation
│   └── ela_analysis.py           # ELA and PCA visualisations
├── Data/
│   └── manifests/
├── figures/                      # All result figures (PNG/JPG)
├── tables/                       # LaTeX result tables
├── results/                      # CSV / TXT metrics output
├── main.tex                      # Beamer presentation (Overleaf-ready)
├── results_data.tex              # LaTeX macros for all metrics
├── README.md
└── .gitignore
```

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/jaini-1801/DeepFake-XGuard.git
cd DeepFake-XGuard
```

### 2. Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Prepare datasets

Download FaceForensics++ and Celeb-DF v2 following their respective access requirements. Update the `ROOT` and `DATA_DIR` paths in each script.

### 4. Run pipeline

```bash
python Src/create_manifests.py
python Src/extract_frames.py
python Src/extract_faces.py
python Src/train_weighted.py
python Src/train_evaluate.py
python Src/evaluate_celebdf.py
python Src/robustness_test.py
python Src/ela_analysis.py
```

---

## Technologies

Python · PyTorch · EfficientNet-B0 · MTCNN · OpenCV · NumPy · Pandas · Matplotlib · Scikit-learn

---

## Limitations

- Performance degrades on unseen datasets (domain shift).
- Severe Gaussian blur collapses AUC to near-chance.
- ELA is sensitive to any post-processing, not only deepfakes.
- PCA provides only a 2D view of a 1280-dimensional space.
- Not evaluated on DFDC or diffusion-based fakes.

---

## Future Work

- Domain adaptation and adversarial training.
- Temporal modelling (RNN / Transformer) over video frames.
- Degradation-robust augmentation pipeline.
- GradCAM saliency maps.
- Evaluation on DFDC, WildDeepfake, and diffusion-generated fakes.

---

## Disclaimer

This project is for academic and research purposes. Its predictions should not be treated as definitive proof that a video or image is authentic or manipulated.

"""
evaluate_celebdf.py
-------------------
Evaluates the trained model on the Celeb-DF dataset (cross-dataset test).

Why is this important?
  The model was trained on FaceForensics++ (FF++). Celeb-DF is a completely
  different dataset with celebrities and higher-quality deepfakes. Testing
  here shows whether the model *generalizes* (works on new data) or just
  memorized FF++ patterns.

Two levels of evaluation:
  1. Frame-level: each face crop is judged independently
  2. Video-level: all face crops from one video are averaged (more reliable)

Outputs:
  - results/celebdf/celebdf_frame_predictions.csv
  - results/celebdf/celebdf_video_predictions.csv
  - Printed metrics for both frame and video levels
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(r"E:\DeepFakeX")

# CSV file listing all Celeb-DF test videos with their labels
MANIFEST   = ROOT / "Data" / "manifests" / "celebdf_official_test.csv"

# Folder with extracted face crops from Celeb-DF test videos
FACES_DIR  = ROOT / "Data" / "faces" / "celebdf_test"

# The best model trained on FaceForensics++
CHECKPOINT = ROOT / "best_efficientnet_b0_weighted.pth"

RESULTS_DIR = ROOT / "results" / "celebdf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
THRESHOLD  = 0.5   # Probability above this -> classified as Fake

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# ==========================================
# IMAGE TRANSFORMATION
# Must match what was used during training (same size and normalization)
# ==========================================

weights = EfficientNet_B0_Weights.DEFAULT

test_transform = transforms.Compose([
    transforms.Resize((160, 160)),       # Same as training
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,  # ImageNet mean (used in EfficientNet training)
        std=weights.transforms().std     # ImageNet std
    )
])

# ==========================================
# DATASET
# Custom dataset that loads Celeb-DF faces using the manifest CSV
# ==========================================

class CelebDFDataset(Dataset):
    """
    Loads face images for Celeb-DF evaluation.

    The manifest CSV has:
      - video_path:  original video file path
      - category:    e.g. 'Celeb-real', 'Celeb-synthesis', 'YouTube-real'
      - label:       0 = Real, 1 = Fake

    We look up each video's extracted face folder and add each face file
    as a sample. Videos with no extracted faces are skipped.

    Returns: (image_tensor, label, video_id, category)
    -- video_id and category are needed for video-level aggregation later.
    """

    def __init__(self, manifest, faces_dir, transform):

        self.df        = pd.read_csv(manifest)
        self.faces_dir = Path(faces_dir)
        self.transform = transform
        self.samples   = []

        for _, row in self.df.iterrows():

            video_path = Path(row["video_path"])
            category   = row["category"]

            # Video ID is the filename without extension (e.g. '00001')
            video_id = video_path.stem
            label    = int(row["label"])

            # Face crops are stored under: faces_dir/{category}/{video_id}/
            video_face_dir = self.faces_dir / category / video_id

            if not video_face_dir.exists():
                continue  # Skip if no faces were extracted for this video

            face_files = sorted(video_face_dir.glob("*.jpg"))

            for face_path in face_files:
                self.samples.append((
                    face_path,
                    label,
                    video_id,
                    category
                ))

        print("Videos in manifest:", len(self.df))
        print("Face images available:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        face_path, label, video_id, category = self.samples[index]

        # Load and transform the face image
        image = Image.open(face_path).convert("RGB")
        image = self.transform(image)

        return image, label, video_id, category


# ---- Create dataset and loader ----
dataset = CelebDFDataset(MANIFEST, FACES_DIR, test_transform)

if len(dataset) == 0:
    raise RuntimeError("No face images found. Check FACES_DIR.")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,     # Keep order consistent for video-level grouping
    num_workers=0
)

# ==========================================
# LOAD MODEL
# Same architecture as training -- must match exactly to load weights
# ==========================================

model = models.efficientnet_b0(weights=None)  # No pretrained weights needed

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2   # 0 = Real, 1 = Fake
)

# Load saved model weights from FF++ training
state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()   # Evaluation mode: disables dropout

print("Weighted EfficientNet-B0 checkpoint loaded.")

# ==========================================
# INFERENCE
# Run all face images through the model and collect predictions
# ==========================================

frame_records = []  # Will hold one record per face image

with torch.no_grad():

    for images, labels, video_ids, categories in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        # Convert raw outputs to probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Probability of being Fake (column 1)
        fake_probs = probabilities[:, 1].cpu().numpy()

        for i in range(len(labels)):
            frame_records.append({
                "video_id":        video_ids[i],
                "category":        categories[i],
                "true_label":      int(labels[i]),
                "fake_probability": float(fake_probs[i])
            })

        # Free GPU memory after each batch
        if DEVICE.type == "cuda":
            del images, outputs, probabilities

print("Inference completed.")

# ==========================================
# FRAME-LEVEL EVALUATION
# Each face image is evaluated independently
# ==========================================

frame_df = pd.DataFrame(frame_records)

# Ground truth labels and predicted probabilities
y_true = frame_df["true_label"].to_numpy()
y_prob = frame_df["fake_probability"].to_numpy()

# Apply threshold: probability >= 0.5 -> Fake (1), else -> Real (0)
y_pred = (y_prob >= THRESHOLD).astype(int)


def print_metrics(name, y_true, y_pred, y_prob):
    """Print a full metrics report for a given set of predictions."""

    print(f"\n========== {name} ==========")

    print("Accuracy:",
          round(accuracy_score(y_true, y_pred), 4))

    print("Precision:",
          round(precision_score(y_true, y_pred, zero_division=0), 4))

    print("Recall:",
          round(recall_score(y_true, y_pred, zero_division=0), 4))

    print("F1 Score:",
          round(f1_score(y_true, y_pred, zero_division=0), 4))

    print("ROC-AUC:",
          round(roc_auc_score(y_true, y_prob), 4))

    print("Balanced Accuracy:",
          round(balanced_accuracy_score(y_true, y_pred), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0
    ))


print_metrics("CELEB-DF FRAME-LEVEL RESULTS", y_true, y_pred, y_prob)

# Save all frame predictions to CSV
frame_df.to_csv(
    RESULTS_DIR / "celebdf_frame_predictions.csv",
    index=False
)

# ==========================================
# VIDEO-LEVEL EVALUATION
# Average all frame predictions for each video, then apply threshold
# This is more robust than frame-level since one bad frame won't ruin the result
# ==========================================

# Group by video: average fake_probability across all frames of that video
video_df = (
    frame_df
    .groupby(["video_id", "category"], as_index=False)
    .agg(
        true_label=("true_label", "first"),          # All frames have the same label
        fake_probability=("fake_probability", "mean"), # Average probability across frames
        detected_faces=("fake_probability", "count")   # How many frames were found
    )
)

video_y_true = video_df["true_label"].to_numpy()
video_y_prob = video_df["fake_probability"].to_numpy()
video_y_pred = (video_y_prob >= THRESHOLD).astype(int)

print_metrics(
    "CELEB-DF VIDEO-LEVEL RESULTS",
    video_y_true,
    video_y_pred,
    video_y_prob
)

# Save video-level predictions to CSV
video_df.to_csv(
    RESULTS_DIR / "celebdf_video_predictions.csv",
    index=False
)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========")

print("Total videos in manifest:", len(pd.read_csv(MANIFEST)))
print("Videos evaluated:", len(video_df))
print("Frames evaluated:", len(frame_df))

print("\nVideo-level category distribution:")
print(video_df.groupby(["category", "true_label"]).size())

print("\nResults saved in:", RESULTS_DIR)

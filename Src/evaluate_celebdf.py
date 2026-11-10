
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

MANIFEST = ROOT / "Data" / "manifests" / "celebdf_official_test.csv"
FACES_DIR = ROOT / "Data" / "faces" / "celebdf_test"
CHECKPOINT = ROOT / "best_efficientnet_b0_weighted.pth"

RESULTS_DIR = ROOT / "results" / "celebdf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# ==========================================
# TRANSFORMATION
# Same as FF++ evaluation
# ==========================================

weights = EfficientNet_B0_Weights.DEFAULT

test_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# ==========================================
# DATASET
# ==========================================

class CelebDFDataset(Dataset):

    def __init__(self, manifest, faces_dir, transform):

        self.df = pd.read_csv(manifest)
        self.faces_dir = Path(faces_dir)
        self.transform = transform

        self.samples = []

        for _, row in self.df.iterrows():

            video_path = Path(row["video_path"])
            category = row["category"]

            video_id = video_path.stem
            label = int(row["label"])

            video_face_dir = (
                self.faces_dir / category / video_id
            )

            if not video_face_dir.exists():
                continue

            face_files = sorted(
                video_face_dir.glob("*.jpg")
            )

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

        face_path, label, video_id, category = (
            self.samples[index]
        )

        image = Image.open(face_path).convert("RGB")
        image = self.transform(image)

        return image, label, video_id, category


dataset = CelebDFDataset(
    MANIFEST,
    FACES_DIR,
    test_transform
)

if len(dataset) == 0:
    raise RuntimeError("No face images found. Check FACES_DIR.")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# LOAD EFFICIENTNET-B0
# ==========================================

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

state_dict = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print("Weighted EfficientNet-B0 checkpoint loaded.")

# ==========================================
# INFERENCE
# ==========================================

frame_records = []

with torch.no_grad():

    for images, labels, video_ids, categories in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        fake_probs = probabilities[:, 1].cpu().numpy()

        for i in range(len(labels)):

            frame_records.append({
                "video_id": video_ids[i],
                "category": categories[i],
                "true_label": int(labels[i]),
                "fake_probability": float(fake_probs[i])
            })

        if DEVICE.type == "cuda":
            del images, outputs, probabilities

print("Inference completed.")

# ==========================================
# FRAME-LEVEL EVALUATION
# ==========================================

frame_df = pd.DataFrame(frame_records)

y_true = frame_df["true_label"].to_numpy()
y_prob = frame_df["fake_probability"].to_numpy()
y_pred = (y_prob >= THRESHOLD).astype(int)

def print_metrics(name, y_true, y_pred, y_prob):

    print(f"\n========== {name} ==========")

    print("Accuracy:",
          round(accuracy_score(y_true, y_pred), 4))

    print("Precision:",
          round(precision_score(
              y_true, y_pred, zero_division=0
          ), 4))

    print("Recall:",
          round(recall_score(
              y_true, y_pred, zero_division=0
          ), 4))

    print("F1 Score:",
          round(f1_score(
              y_true, y_pred, zero_division=0
          ), 4))

    print("ROC-AUC:",
          round(roc_auc_score(y_true, y_prob), 4))

    print("Balanced Accuracy:",
          round(balanced_accuracy_score(
              y_true, y_pred
          ), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0
    ))


print_metrics(
    "CELEB-DF FRAME-LEVEL RESULTS",
    y_true,
    y_pred,
    y_prob
)

frame_df.to_csv(
    RESULTS_DIR / "celebdf_frame_predictions.csv",
    index=False
)

# ==========================================
# VIDEO-LEVEL EVALUATION
# Average face probabilities per video
# ==========================================

video_df = (
    frame_df
    .groupby(["video_id", "category"], as_index=False)
    .agg(
        true_label=("true_label", "first"),
        fake_probability=("fake_probability", "mean"),
        detected_faces=("fake_probability", "count")
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
print(video_df.groupby(
    ["category", "true_label"]
).size())

print("\nResults saved in:", RESULTS_DIR)

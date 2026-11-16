"""
robustness_test_robust.py
-------------------------
Evaluates the degradation-aware EfficientNet-B0 checkpoint on the FF++
test split under clean, JPEG, resize, and blur conditions.

Labels are assigned from the immediate class folder under DATA_DIR:
  original/ -> real (0)
  every manipulation folder -> fake (1)

The script stops with a useful error if either class is missing. This
prevents misleading F1=0 / ROC-AUC=NaN reports caused by a one-class test set.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
)
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

# Resolve paths relative to this repository, not a particular Windows drive.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data" / "faces" / "test"
CHECKPOINT = ROOT / "best_efficientnet_b0_robust.pth"
OUTPUT = ROOT / "results" / "robust_model_robustness.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5
BATCH_SIZE = 8

print("Device:", DEVICE)
print("Test data:", DATA_DIR)
print("Checkpoint:", CHECKPOINT)

if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"Test face directory not found: {DATA_DIR}. "
        "Run the frame/face extraction pipeline first."
    )
if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}. "
        "Run Src/train_robust.py before this evaluation."
    )

# Load image paths and labels. Folder names directly below DATA_DIR define
# the classes; nested video-ID folders are searched recursively.
samples = []
class_counts = {"real": 0, "fake": 0}
image_extensions = {".jpg", ".jpeg", ".png"}

for class_dir in sorted(DATA_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    label = 0 if class_dir.name.lower() == "original" else 1
    for path in class_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            samples.append((path, label))
            class_counts["real" if label == 0 else "fake"] += 1

print("Test images:", len(samples))
print("Real images:", class_counts["real"])
print("Fake images:", class_counts["fake"])

if not samples or class_counts["real"] == 0 or class_counts["fake"] == 0:
    raise RuntimeError(
        "The test split must contain both real and fake images. "
        f"Observed counts: {class_counts}. "
        "Check that DATA_DIR points to data/faces/test and that it contains "
        "an 'original' folder plus the manipulation folders."
    )

# The checkpoint already contains the trained weights. weights=None avoids
# an unnecessary ImageNet download before loading that checkpoint.
weights = EfficientNet_B0_Weights.DEFAULT
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])

def degrade(image, condition, severity):
    """Apply one controlled image degradation before model preprocessing."""
    if condition == "clean":
        return image

    if condition == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()

    if condition == "resize":
        width, height = image.size
        factor = float(severity)
        small = image.resize(
            (max(1, int(width * factor)), max(1, int(height * factor))),
            Image.Resampling.BILINEAR,
        )
        return small.resize((width, height), Image.Resampling.BILINEAR)

    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(severity)))

    raise ValueError(f"Unknown degradation condition: {condition}")

# Rebuild the same EfficientNet-B0 classification head used during training.
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
state_dict = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("Degradation-aware checkpoint loaded.")

conditions = [
    ("clean", 0),
    ("jpeg", 90), ("jpeg", 70), ("jpeg", 50), ("jpeg", 30),
    ("resize", 0.75), ("resize", 0.50), ("resize", 0.25),
    ("blur", 1), ("blur", 2), ("blur", 3), ("blur", 4),
]

results = []
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

for condition, severity in conditions:
    labels_all = []
    probabilities_all = []

    # Batch inference keeps the evaluation faster while limiting memory use.
    for start in range(0, len(samples), BATCH_SIZE):
        batch_samples = samples[start:start + BATCH_SIZE]
        tensors = []
        batch_labels = []

        for path, label in batch_samples:
            with Image.open(path) as image_file:
                image = image_file.convert("RGB")
            image = degrade(image, condition, severity)
            tensors.append(transform(image))
            batch_labels.append(label)

        batch = torch.stack(tensors).to(DEVICE)

        with torch.inference_mode():
            logits = model(batch)
            fake_probabilities = torch.softmax(logits, dim=1)[:, 1]

        labels_all.extend(batch_labels)
        probabilities_all.extend(fake_probabilities.cpu().numpy().tolist())

    y_true = np.asarray(labels_all, dtype=np.int64)
    y_prob = np.asarray(probabilities_all, dtype=np.float64)

    # Defensive check: AUC is undefined when the evaluation set has one class.
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise RuntimeError(
            f"Both labels were not present for condition {condition}/{severity}."
        )

    y_pred = (y_prob >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    row = {
        "condition": condition,
        "severity": severity,
        "threshold": THRESHOLD,
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "fake_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    results.append(row)

    # Save after every condition so completed work survives an interruption.
    pd.DataFrame(results).to_csv(OUTPUT, index=False)
    print(
        f"{condition:6} severity={severity}: "
        f"F1={row['f1']:.4f} | AUC={row['roc_auc']:.4f} | "
        f"Balanced Acc={row['balanced_acc']:.4f}"
    )

print("\nEvaluation completed.")
print("Results saved to:", OUTPUT)

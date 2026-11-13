"""
robustness_test_robust.py
-------------------------
Tests the ROBUST model (trained with DegradationAugmentation in train_robust.py)
against the same set of degradations used in robustness_test.py.

Purpose:
  Compare the robust model vs the baseline model to see if
  degradation-aware training actually improved resilience to
  JPEG compression, downscaling, and blurring.

Processes images ONE AT A TIME (not in batches) to keep memory usage low.
Saves results to CSV after each condition so progress is not lost on crash.

Output:
  results/robust_model_robustness.csv
  Columns: condition, severity, f1, roc_auc, balanced_acc, specificity, recall
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image, ImageFilter
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix
)

# ---- Config ----
ROOT     = Path(r"E:\DeepFakeX")
DATA_DIR = ROOT / "data" / "faces" / "test"

# The robust model (trained with DegradationAugmentation)
CHECKPOINT = ROOT / "best_efficientnet_b0_robust.pth"
OUTPUT     = ROOT / "results" / "robust_model_robustness.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESH = 0.5   # Decision threshold: probability >= THRESH -> Fake

print("Device:", DEVICE)

# ---- Image preprocessing (same as training) ----
weights = EfficientNet_B0_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# ---- Load all test image paths ----
samples = []

for folder in DATA_DIR.iterdir():
    if folder.is_dir():
        # 'original' = Real (0), all other folders = Fake (1)
        label = 0 if folder.name == "original" else 1

        for path in folder.rglob("*.jpg"):
            samples.append((path, label))

print("Total images:", len(samples))
print("Real:", sum(y == 0 for _, y in samples))
print("Fake:", sum(y == 1 for _, y in samples))


# ---- Degradation function ----
def degrade(image, condition, severity):
    """
    Apply a specific degradation to a PIL image.

    Args:
        image:     PIL Image to modify
        condition: 'jpeg', 'resize', or 'blur' (or anything else = no change)
        severity:  strength of degradation:
                   - jpeg:   JPEG quality (e.g. 30 = very compressed)
                   - resize: scale factor (e.g. 0.25 = 25% of original size)
                   - blur:   Gaussian radius (e.g. 4 = very blurry)

    Returns: degraded PIL Image
    """

    if condition == "jpeg":
        # Re-compress the image at the given JPEG quality
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)
        with Image.open(buffer) as temp:
            image = temp.convert("RGB")

    elif condition == "resize":
        # Shrink to `severity` fraction of original size, then enlarge back
        w, h    = image.size
        factor  = float(severity)
        small   = image.resize(
            (max(1, int(w * factor)), max(1, int(h * factor))),
            Image.Resampling.BILINEAR
        )
        image = small.resize((w, h), Image.Resampling.BILINEAR)

    elif condition == "blur":
        # Apply Gaussian blur with the given radius
        image = image.filter(
            ImageFilter.GaussianBlur(radius=float(severity))
        )

    return image  # For 'clean', no change is made (never reaches this in conditions)


# ---- Load robust model ----
model = models.efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint)
model = model.to(DEVICE)
model.eval()

print("Robust model loaded successfully.")


# ---- Define conditions to test ----
# Same list as robustness_test.py for a fair comparison
conditions = [
    ("clean",  0),
    ("jpeg",  90),  ("jpeg",  70), ("jpeg",  50), ("jpeg",  30),
    ("resize", 0.75), ("resize", 0.5), ("resize", 0.25),
    ("blur",   1),  ("blur",   2),  ("blur",   3),  ("blur",   4),
]

results = []  # Accumulates one result dict per condition


# ---- Evaluate each condition ----
for condition, severity in conditions:

    print(f"\nTesting: {condition}, severity={severity}")

    labels_all = []
    probs_all  = []

    with torch.inference_mode():

        # Process images ONE AT A TIME (saves RAM vs batch processing)
        for index, (path, label) in enumerate(samples):

            with Image.open(path) as im:
                image = im.convert("RGB")

            # Apply degradation to raw image first
            image = degrade(image, condition, severity)

            # Preprocess: resize + normalize, add batch dimension
            tensor = transform(image).unsqueeze(0).to(DEVICE)

            output = model(tensor)

            # Get probability of being Fake (class index 1)
            probability = torch.softmax(output, dim=1)[0, 1].item()

            labels_all.append(label)
            probs_all.append(probability)

            del tensor, output  # Free GPU/CPU memory immediately

            # Print progress every 500 images
            if (index + 1) % 500 == 0:
                print(f"Processed {index + 1}/{len(samples)}")

    y    = np.array(labels_all)
    p    = np.array(probs_all)
    pred = (p >= THRESH).astype(int)  # Apply threshold to get class predictions

    # Confusion matrix for computing specificity
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    # Specificity = True Negative Rate = how well we identify REAL images
    specificity = tn / (tn + fp) if (tn + fp) else 0

    f1           = f1_score(y, pred, zero_division=0)
    auc          = roc_auc_score(y, p)
    balanced_acc = balanced_accuracy_score(y, pred)
    recall_val   = recall_score(y, pred, zero_division=0)  # Fake recall (sensitivity)

    results.append({
        "condition":    condition,
        "severity":     severity,
        "f1":           f1,
        "roc_auc":      auc,
        "balanced_acc": balanced_acc,
        "specificity":  specificity,
        "recall":       recall_val
    })

    # Save after every condition so progress is not lost if the script crashes
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT, index=False)

    print(
        f"F1={f1:.4f} | AUC={auc:.4f} | "
        f"Balanced Acc={balanced_acc:.4f} | "
        f"Specificity={specificity:.4f}"
    )

print("\nEvaluation completed.")
print("Results saved to:", OUTPUT)

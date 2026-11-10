
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

# ---------------- CONFIG ----------------

ROOT = Path(r"E:\DeepFakeX")
DATA_DIR = ROOT / "data" / "faces" / "test"

CHECKPOINT = ROOT / "best_efficientnet_b0_robust.pth"
OUTPUT = ROOT / "results" / "robust_model_robustness.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESH = 0.5

print("Device:", DEVICE)

# ---------------- TRANSFORM ----------------

weights = EfficientNet_B0_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# ---------------- DATASET ----------------

samples = []

for folder in DATA_DIR.iterdir():
    if folder.is_dir():
        label = 0 if folder.name == "original" else 1

        for path in folder.rglob("*.jpg"):
            samples.append((path, label))

print("Total images:", len(samples))
print("Real:", sum(y == 0 for _, y in samples))
print("Fake:", sum(y == 1 for _, y in samples))

# ---------------- DEGRADATION ----------------

def degrade(image, condition, severity):

    if condition == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)

        with Image.open(buffer) as temp:
            image = temp.convert("RGB")

    elif condition == "resize":
        w, h = image.size
        factor = float(severity)

        small = image.resize(
            (max(1, int(w * factor)),
             max(1, int(h * factor))),
            Image.Resampling.BILINEAR
        )

        image = small.resize(
            (w, h),
            Image.Resampling.BILINEAR
        )

    elif condition == "blur":
        image = image.filter(
            ImageFilter.GaussianBlur(radius=float(severity))
        )

    return image

# ---------------- MODEL ----------------

model = models.efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model = model.to(DEVICE)
model.eval()

print("Robust model loaded successfully.")

# ---------------- CONDITIONS ----------------

conditions = [
    ("clean", 0),
    ("jpeg", 90),
    ("jpeg", 70),
    ("jpeg", 50),
    ("jpeg", 30),
    ("resize", 0.75),
    ("resize", 0.5),
    ("resize", 0.25),
    ("blur", 1),
    ("blur", 2),
    ("blur", 3),
    ("blur", 4)
]

results = []

# ---------------- EVALUATION ----------------

for condition, severity in conditions:

    print(f"\nTesting: {condition}, severity={severity}")

    labels_all = []
    probs_all = []

    with torch.inference_mode():

        # Process ONE image at a time to reduce RAM usage
        for index, (path, label) in enumerate(samples):

            with Image.open(path) as im:
                image = im.convert("RGB")

            image = degrade(image, condition, severity)

            tensor = transform(image).unsqueeze(0).to(DEVICE)

            output = model(tensor)
            probability = torch.softmax(output, dim=1)[0, 1].item()

            labels_all.append(label)
            probs_all.append(probability)

            del tensor, output

            if (index + 1) % 500 == 0:
                print(f"Processed {index + 1}/{len(samples)}")

    y = np.array(labels_all)
    p = np.array(probs_all)
    pred = (p >= THRESH).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    f1 = f1_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, p)
    balanced_acc = balanced_accuracy_score(y, pred)
    recall = recall_score(y, pred, zero_division=0)

    results.append({
        "condition": condition,
        "severity": severity,
        "f1": f1,
        "roc_auc": auc,
        "balanced_acc": balanced_acc,
        "specificity": specificity,
        "recall": recall
    })

    # Save after every condition
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        OUTPUT, index=False
    )

    print(
        f"F1={f1:.4f} | AUC={auc:.4f} | "
        f"Balanced Acc={balanced_acc:.4f} | "
        f"Specificity={specificity:.4f}"
    )

print("\nEvaluation completed.")
print("Results saved to:", OUTPUT)

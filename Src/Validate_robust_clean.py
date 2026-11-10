
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score
)

# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(r"E:\DeepFakeX")

VAL_DATA = ROOT / "data" / "faces" / "val"
TEST_DATA = ROOT / "data" / "faces" / "test"

# Currently evaluating the robust model
CKPT = ROOT / "best_efficientnet_b0.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

checkpoint = torch.load(
    CKPT,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")

# --------------------------------------------------
# 3. IMAGE TRANSFORMATION
# --------------------------------------------------

weights = EfficientNet_B0_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# --------------------------------------------------
# 4. LOAD IMAGE PATHS
# --------------------------------------------------

def load_samples(folder):

    samples = []

    for class_folder in sorted(folder.iterdir()):

        if not class_folder.is_dir():
            continue

        label = 0 if class_folder.name == "original" else 1

        for extension in ("*.jpg", "*.jpeg", "*.png"):
            samples.extend(
                (p, label)
                for p in class_folder.rglob(extension)
            )

    return samples


# --------------------------------------------------
# 5. GET PREDICTION PROBABILITIES
# --------------------------------------------------

def evaluate_dataset(samples, dataset_name):

    y_true = []
    y_prob = []

    with torch.inference_mode():

        for i, (path, label) in enumerate(samples):

            try:
                with Image.open(path) as im:
                    image = transform(
                        im.convert("RGB")
                    ).unsqueeze(0).to(DEVICE)

                output = model(image)

                probability = torch.softmax(
                    output, dim=1
                )[0, 1].item()

                y_true.append(label)
                y_prob.append(probability)

            except Exception as e:
                print("Error reading:", path, e)

            if (i + 1) % 500 == 0:
                print(
                    f"{dataset_name}: "
                    f"Processed {i+1}/{len(samples)}"
                )

    return np.array(y_true), np.array(y_prob)


# --------------------------------------------------
# 6. LOAD VALIDATION DATA
# --------------------------------------------------

val_samples = load_samples(VAL_DATA)

print("\nValidation images:", len(val_samples))
print("Validation labels:", Counter(y for _, y in val_samples))

val_true, val_prob = evaluate_dataset(
    val_samples, "Validation"
)

if len(val_true) != len(val_samples):
    raise RuntimeError("Some validation images failed to load.")

# --------------------------------------------------
# 7. FIND BEST THRESHOLD ON VALIDATION DATA
# --------------------------------------------------

best_threshold = 0.5
best_score = -1

for threshold in np.arange(0.10, 0.91, 0.01):

    predictions = (val_prob >= threshold).astype(int)

    score = balanced_accuracy_score(
        val_true,
        predictions
    )

    if score > best_score:
        best_score = score
        best_threshold = float(threshold)

print("\nVALIDATION RESULTS")
print("----------------------------")
print("Best threshold:", best_threshold)
print("Best balanced accuracy:", best_score)

# --------------------------------------------------
# 8. LOAD TEST DATA
# --------------------------------------------------

test_samples = load_samples(TEST_DATA)

print("\nTest images:", len(test_samples))
print("Test labels:", Counter(y for _, y in test_samples))

test_true, test_prob = evaluate_dataset(
    test_samples, "Test"
)

if len(test_true) != len(test_samples):
    raise RuntimeError("Some test images failed to load.")

# --------------------------------------------------
# 9. TEST USING FIXED VALIDATION THRESHOLD
# --------------------------------------------------

test_pred = (test_prob >= best_threshold).astype(int)

cm = confusion_matrix(
    test_true,
    test_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nFINAL TEST RESULTS")
print("============================")
print("Threshold:", best_threshold)

print("Accuracy:",
      accuracy_score(test_true, test_pred))

print("Precision:",
      precision_score(test_true, test_pred, zero_division=0))

print("Fake Recall:",
      recall_score(test_true, test_pred, zero_division=0))

print("F1 Score:",
      f1_score(test_true, test_pred, zero_division=0))

print("ROC-AUC:",
      roc_auc_score(test_true, test_prob))

print("Balanced Accuracy:",
      balanced_accuracy_score(test_true, test_pred))

print("Specificity (Real Recall):",
      tn / (tn + fp) if (tn + fp) > 0 else 0)

print("\nConfusion Matrix:")
print(cm)

print("\nPredicted labels:")
print(Counter(test_pred))

# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

output_file = ROOT / "results" / "threshold_tuned_results.txt"

output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w") as f:

    f.write(f"Checkpoint: {CKPT}\n")
    f.write(f"Threshold: {best_threshold}\n")
    f.write(f"Validation balanced accuracy: {best_score}\n")
    f.write(f"Test accuracy: {accuracy_score(test_true, test_pred)}\n")
    f.write(f"Test precision: {precision_score(test_true, test_pred, zero_division=0)}\n")
    f.write(f"Test recall: {recall_score(test_true, test_pred, zero_division=0)}\n")
    f.write(f"Test F1: {f1_score(test_true, test_pred, zero_division=0)}\n")
    f.write(f"Test ROC-AUC: {roc_auc_score(test_true, test_prob)}\n")
    f.write(f"Test balanced accuracy: {balanced_accuracy_score(test_true, test_pred)}\n")
    f.write(f"Confusion matrix:\n{cm}\n")

print("\nResults saved to:", output_file)

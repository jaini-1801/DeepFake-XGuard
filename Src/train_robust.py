"""
train_robust.py
---------------
Training script with degradation-aware augmentation for robustness.

Problem: Real-world deepfake videos are often compressed, blurry, or
low-resolution. A model trained on clean high-quality images may fail
on these degraded inputs.

Solution applied here (DegradationAugmentation):
  During training, randomly apply one of these degradations to each image:
    - JPEG compression (quality 30-90): simulates social media compression
    - Downscaling (25-75% then upscaled): simulates low-resolution video
    - Gaussian blur (radius 0.5-2.0): simulates out-of-focus or motion blur
    - No degradation (25% of the time): keep some clean examples

This forces the model to learn features that survive these distortions.
Best model is saved based on F1 score.
"""

import io
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image, ImageFilter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ---- Config ----
ROOT        = Path(r"E:\DeepFakeX")
DATA_DIR    = ROOT / "data" / "faces"
RESULTS_DIR = ROOT / "results"
CHECKPOINT  = ROOT / "best_efficientnet_b0_robust.pth"  # Where to save best model

BATCH_SIZE = 8
EPOCHS     = 10
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# Get the standard ImageNet mean/std used by EfficientNet
weights = EfficientNet_B0_Weights.DEFAULT
mean = weights.transforms().mean
std  = weights.transforms().std


class DegradationAugmentation:
    """
    Randomly applies a real-world image degradation to simulate
    how deepfakes look after compression or transmission.

    One of four options is randomly chosen each call:
      - 'jpeg'   : re-save at a random low quality (simulates compression artifacts)
      - 'resize' : shrink then enlarge (simulates pixelation from low resolution)
      - 'blur'   : apply Gaussian blur (simulates camera blur or motion)
      - 'none'   : return image unchanged (keeps some clean training examples)
    """

    def __call__(self, image):
        choice = random.choice(["jpeg", "resize", "blur", "none"])

        if choice == "jpeg":
            # Re-compress the image at a random quality between 30 and 90
            # Lower quality = more compression artifacts (blocky edges, color shifts)
            quality = random.randint(30, 90)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")

        elif choice == "resize":
            # Shrink the image to 25%, 50%, or 75% of original size, then enlarge back
            # This creates pixelation/blurriness similar to low-resolution video
            factor = random.choice([0.25, 0.5, 0.75])
            w, h = image.size
            small = image.resize(
                (max(1, int(w * factor)), max(1, int(h * factor))),
                Image.Resampling.BILINEAR
            )
            image = small.resize((w, h), Image.Resampling.BILINEAR)

        elif choice == "blur":
            # Apply Gaussian blur with a random radius between 0.5 and 2.0
            # Simulates out-of-focus or motion blur
            image = image.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0))
            )

        return image  # 'none' case: return unchanged


# ---- Training transforms ----
# Apply random degradation FIRST, then standard augmentation + normalization
train_transform = transforms.Compose([
    DegradationAugmentation(),              # Random real-world distortion
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Evaluation transforms: no degradation, just resize + normalize
eval_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


class BinaryFaceDataset(Dataset):
    """
    Loads face images from folder structure:
        root/
          original/   -> Real (label = 0)
          Deepfakes/  -> Fake (label = 1)
          ...
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for class_dir in self.root.iterdir():
            if not class_dir.is_dir():
                continue

            label = 0 if class_dir.name == "original" else 1

            # Collect all jpg files from this class folder
            self.samples.extend(
                (path, label) for path in class_dir.rglob("*.jpg")
            )

        if not self.samples:
            raise RuntimeError(f"No JPG images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        # Open with 'with' to properly close the file handle
        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ---- Load datasets ----
train_dataset = BinaryFaceDataset(DATA_DIR / "train", train_transform)
val_dataset   = BinaryFaceDataset(DATA_DIR / "val",   eval_transform)
test_dataset  = BinaryFaceDataset(DATA_DIR / "test",  eval_transform)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# ---- Create data loaders ----
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0
)
val_loader = DataLoader(
    val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ---- Build model ----
# EfficientNet-B0 with final layer replaced for binary classification
model = models.efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

# Standard cross-entropy loss (no class weighting here, unlike train_weighted.py)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=1e-4
)


def evaluate(loader):
    """
    Evaluate the model on a data loader.
    Returns: (accuracy, precision, recall, f1, roc_auc)
    """
    model.eval()
    labels_all, preds_all, probs_all = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            labels_all.extend(labels.numpy())
            preds_all.extend(probs.argmax(dim=1).cpu().numpy())  # Predicted class
            probs_all.extend(probs[:, 1].cpu().numpy())          # Fake probability

    return (
        accuracy_score(labels_all, preds_all),
        precision_score(labels_all, preds_all, zero_division=0),
        recall_score(labels_all, preds_all, zero_division=0),
        f1_score(labels_all, preds_all, zero_division=0),
        roc_auc_score(labels_all, probs_all),
    )


# ---- Training loop ----
best_f1 = -1.0  # Track best validation F1
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    metrics = evaluate(val_loader)
    accuracy, precision, recall, f1, auc = metrics

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss {running_loss / len(train_loader):.4f} | "
        f"Val F1 {f1:.4f} | Val AUC {auc:.4f}"
    )

    # Save checkpoint whenever F1 improves
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), CHECKPOINT)
        print("Saved robust checkpoint:", CHECKPOINT)

# ---- Final test evaluation ----
print("\nLoading best robust checkpoint...")
model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
)

accuracy, precision, recall, f1, auc = evaluate(test_loader)

# Format and save the results report
report = (
    "Degradation-aware training -- clean test set\n"
    f"Accuracy : {accuracy:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall   : {recall:.4f}\n"
    f"F1 Score : {f1:.4f}\n"
    f"ROC-AUC  : {auc:.4f}\n"
)

print("\n" + report)
(RESULTS_DIR / "robust_clean_metrics.txt").write_text(report)
print("Metrics saved to:", RESULTS_DIR / "robust_clean_metrics.txt")

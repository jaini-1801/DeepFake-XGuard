"""
evaluate.py
-----------
Evaluates a trained EfficientNet-B0 model on the test split of FaceForensics++.

What it does:
  1. Loads face images from the test folder
  2. Loads the best saved model checkpoint
  3. Runs inference (prediction) on all test images
  4. Computes accuracy, precision, recall, F1, ROC-AUC
  5. Saves a confusion matrix image and a metrics text file
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ---- Paths ----
ROOT = Path(r"E:\DeepFakeX")
DATA_DIR = ROOT / "data" / "faces"          # Folder containing extracted face images
CHECKPOINT = ROOT / "best_efficientnet_b0.pth"  # Saved model weights
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)  # Create results folder if it doesn't exist

# Use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4  # Small batch size to be safe on memory

print("Device:", DEVICE)


class BinaryFaceDataset(Dataset):
    """
    Loads face images from a folder structure:
        root/
          original/   -> Real faces  (label = 0)
          Deepfakes/  -> Fake faces  (label = 1)
          ...

    Supports .jpg, .jpeg, .png image formats.
    Raises an error if no images are found (avoids silent failures).
    """

    def __init__(self, root, transform=None):
        self.samples = []       # List of (image_path, label) tuples
        self.transform = transform

        for folder in Path(root).iterdir():
            if not folder.is_dir():
                continue

            # 'original' = real face (0); anything else = fake (1)
            label = 0 if folder.name == "original" else 1

            for path in folder.rglob("*"):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((path, label))

        # Safety check: crash early if the folder is empty or path is wrong
        if not self.samples:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Open image safely (auto-closes file handle after the 'with' block)
        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ---- Image preprocessing ----
# Use the same normalization stats the model was trained with (ImageNet stats)
weights = EfficientNet_B0_Weights.DEFAULT
transform = transforms.Compose([
    transforms.Resize((160, 160)),       # Resize to 160x160 (same as training)
    transforms.ToTensor(),               # Convert image to tensor
    transforms.Normalize(                # Normalize pixel values
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# ---- Load test dataset ----
test_dataset = BinaryFaceDataset(
    DATA_DIR / "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,     # Keep order consistent for reproducible evaluation
    num_workers=0
)

print("Test images:", len(test_dataset))

# ---- Load model ----
# Check the checkpoint exists before trying to load it
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

# Rebuild the model architecture (must match what was trained)
model = models.efficientnet_b0(weights=None)  # No pretrained weights needed here
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2   # 2 classes: Real and Fake
)

# Load saved weights into the model
state = torch.load(
    CHECKPOINT, map_location=DEVICE, weights_only=True
)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()  # Put model in eval mode (turns off dropout, etc.)

# ---- Run inference ----
all_labels = []   # True labels (0 = Real, 1 = Fake)
all_preds = []    # Model's predicted class
all_probs = []    # Model's predicted probability of being Fake

with torch.no_grad():  # Disable gradient computation (saves memory, speeds up)
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)

        # Convert raw outputs to probabilities using softmax
        probs = torch.softmax(outputs, dim=1)

        # Pick the class with the highest probability
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())  # Fake class probability

# ---- Compute metrics ----
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall    = recall_score(all_labels, all_preds, zero_division=0)
f1        = f1_score(all_labels, all_preds, zero_division=0)
auc       = roc_auc_score(all_labels, all_probs)

# Build confusion matrix: shows TP, TN, FP, FN counts
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

print("\nConfusion Matrix (rows=true, columns=predicted):")
print("                 Predicted Real   Predicted Fake")
print(f"Actual Real          {cm[0, 0]:<12} {cm[0, 1]}")
print(f"Actual Fake          {cm[1, 0]:<12} {cm[1, 1]}")

# ---- Save confusion matrix as image ----
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real", "Fake"]
)
disp.plot()
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=300)
plt.close()

# ---- Save metrics to a text file ----
metrics = (
    f"Accuracy: {accuracy:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall: {recall:.4f}\n"
    f"F1-score: {f1:.4f}\n"
    f"ROC-AUC: {auc:.4f}\n\n"
    f"Confusion matrix:\n{cm}\n"
)

(RESULTS / "baseline_metrics.txt").write_text(
    metrics, encoding="utf-8"
)

print("\n===== TEST RESULTS =====")
print(metrics)
print("Saved confusion matrix:", RESULTS / "confusion_matrix.png")
print("Saved metrics:", RESULTS / "baseline_metrics.txt")

"""
train.py
--------
Main training script for the deepfake detector.
Uses EfficientNet-B0 (a lightweight but powerful image classification model)
to tell apart REAL faces from FAKE/manipulated ones.

Flow:
  1. Load face images from train/val/test folders
  2. Apply data augmentation (flips, rotations) to make the model more robust
  3. Train for EPOCHS rounds, saving the best model based on F1 score
  4. Run final evaluation on the test set and print results
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights
from torchvision import transforms, models
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from pathlib import Path

# =========================
# CONFIG
# All important settings in one place -- change these to fit your setup
# =========================

ROOT = Path(r"E:\DeepFakeX")            # Root folder of the whole project
DATA_DIR = Path(r"E:\DeepFakeX\data\faces")  # Where face images are stored

BATCH_SIZE = 8      # How many images to process at once (lower = less RAM)
EPOCHS = 10         # How many full passes through the training data
LR = 1e-4           # Learning rate -- how fast the model updates its weights
NUM_WORKERS = 0     # Number of background threads for loading data (0 = main thread)

# Use GPU if available, otherwise fall back to CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# =========================
# TRANSFORMS
# Image preprocessing -- resize + normalize so images match what EfficientNet expects
# =========================

# Load the default pretrained weights for EfficientNet-B0
weights = EfficientNet_B0_Weights.DEFAULT

# Training transforms: include random flips and small rotations for variety
train_transform = transforms.Compose([
    transforms.Resize((160, 160)),          # Shrink image to 160x160 pixels
    transforms.RandomHorizontalFlip(),      # Randomly mirror image left-right
    transforms.RandomRotation(5),           # Randomly tilt image up to 5 degrees
    transforms.ToTensor(),                  # Convert PIL image to PyTorch tensor
    transforms.Normalize(                   # Scale pixel values to match ImageNet stats
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# Validation/test transforms: no augmentation, just resize and normalize
val_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


class BinaryFaceDataset(Dataset):
    """
    Custom dataset that loads face images from a folder structure like:
        root/
          original/   <- real faces, label = 0
          Deepfakes/  <- fake faces, label = 1
          Face2Face/  <- fake faces, label = 1
          ...

    Scans all subfolders; anything named 'original' is labeled Real (0),
    everything else is labeled Fake (1).
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # Will hold (image_path, label) pairs

        for class_dir in self.root.iterdir():
            if not class_dir.is_dir():
                continue

            # 'original' folder = Real faces (label 0)
            # All other folders = Fake faces (label 1)
            label = 0 if class_dir.name == "original" else 1

            # Collect all .jpg files inside this class folder (recursively)
            for img_path in class_dir.rglob("*.jpg"):
                self.samples.append((img_path, label))

    def __len__(self):
        # Total number of images in this dataset
        return len(self.samples)

    def __getitem__(self, idx):
        # Load one image and its label by index
        img_path, label = self.samples[idx]

        # Open image and ensure it's RGB (not grayscale or RGBA)
        image = Image.open(img_path).convert("RGB")

        # Apply transforms (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# DATASETS
# Create separate dataset objects for train, validation, and test splits
# =========================

train_dataset = BinaryFaceDataset(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = BinaryFaceDataset(
    DATA_DIR / "val",
    transform=val_transform
)

test_dataset = BinaryFaceDataset(
    DATA_DIR / "test",
    transform=val_transform
)

# Print how many images are in each split
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")


# =========================
# DATALOADERS
# DataLoaders wrap datasets and serve batches of images during training
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       # Shuffle training data each epoch to prevent memorization
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,      # No need to shuffle for evaluation
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# =========================
# MODEL
# Load EfficientNet-B0 pretrained on ImageNet, then replace the last layer
# to output 2 classes (Real vs Fake) instead of 1000 ImageNet classes
# =========================

model = models.efficientnet_b0(weights=weights)

# Replace the final classification layer: 1280 features -> 2 output classes
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2   # 0 = Real, 1 = Fake
)

# Move model to GPU or CPU
model = model.to(DEVICE)


# =========================
# LOSS + OPTIMIZER
# CrossEntropyLoss: measures how wrong the model's predictions are
# AdamW: optimizer that adjusts model weights to minimize the loss
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4   # L2 regularization to prevent overfitting
)


# =========================
# EVALUATION FUNCTION
# Runs the model on a dataset without updating weights, returns metrics
# =========================

def evaluate(model, loader):
    """
    Evaluate the model on a given dataset.
    Returns: accuracy, precision, recall, F1 score, and ROC-AUC.

    - Accuracy: % of correct predictions
    - Precision: of all predicted fakes, how many are actually fake
    - Recall: of all real fakes, how many did we catch
    - F1: balance between precision and recall
    - ROC-AUC: overall ranking ability of the model (1.0 = perfect)
    """

    model.eval()  # Switch to evaluation mode (disables dropout etc.)

    all_labels = []  # Ground truth labels
    all_preds = []   # Predicted class (0 or 1)
    all_probs = []   # Predicted probability of being fake

    with torch.no_grad():  # Don't compute gradients (saves memory)

        for images, labels in loader:

            images = images.to(DEVICE)

            # Forward pass: get raw scores for each class
            outputs = model(images)

            # Convert raw scores to probabilities (sum to 1.0)
            probabilities = torch.softmax(outputs, dim=1)

            # Pick the class with the highest probability
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(predictions.cpu().numpy())

            # Probability of being Fake (class index 1)
            all_probs.extend(
                probabilities[:, 1].cpu().numpy()
            )

            # Build and save the confusion matrix after each batch
            # (shows TN, FP, FN, TP counts)
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

            print("Confusion Matrix:")
            print(cm)

            # Save confusion matrix image to disk
            results_dir = Path(r"E:\DeepFakeX\results")
            results_dir.mkdir(parents=True, exist_ok=True)

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Real", "Fake"]
            )

            disp.plot()
            plt.tight_layout()
            plt.savefig(results_dir / "confusion_matrix.png", dpi=300)
            plt.close()
            print("Saved confusion matrix.")

    # Calculate all metrics using the collected predictions
    accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(
        all_labels, all_preds, zero_division=0
    )

    recall = recall_score(
        all_labels, all_preds, zero_division=0
    )

    f1 = f1_score(
        all_labels, all_preds, zero_division=0
    )

    auc = roc_auc_score(all_labels, all_probs)

    return accuracy, precision, recall, f1, auc


# =========================
# TRAINING LOOP
# For each epoch: train on all batches, then evaluate on validation set
# Save the model whenever it beats the previous best F1 score
# =========================

best_f1 = 0.0  # Track the best F1 seen so far

for epoch in range(EPOCHS):

    model.train()  # Switch to training mode (enables dropout, batch norm updates)

    running_loss = 0.0  # Accumulate loss across all batches this epoch

    for images, labels in train_loader:

        # Move data to GPU/CPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()   # Clear gradients from the previous step

        outputs = model(images) # Forward pass

        # Compute how wrong the predictions are
        loss = criterion(outputs, labels)

        loss.backward()         # Backpropagate: compute gradients

        optimizer.step()        # Update model weights

        running_loss += loss.item()

    # Average loss over all batches this epoch
    train_loss = running_loss / len(train_loader)

    # Evaluate on validation set after each epoch
    metrics = evaluate(model, val_loader)
    accuracy, precision, recall, f1, auc = metrics

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Loss: {train_loss:.4f}")
    print(f"Val Accuracy: {accuracy:.4f}")
    print(f"Val Precision: {precision:.4f}")
    print(f"Val Recall: {recall:.4f}")
    print(f"Val F1: {f1:.4f}")
    print(f"Val ROC-AUC: {auc:.4f}")

    # Save the model if this epoch has the best F1 so far
    if f1 > best_f1:
        best_f1 = f1
        torch.save(
            model.state_dict(),
            ROOT / "best_efficientnet_b0.pth"
        )
        print("✓ Best model saved")


# =========================
# FINAL TEST EVALUATION
# Load the best saved model and run on the test set (never seen during training)
# =========================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        ROOT / "best_efficientnet_b0.pth",
        map_location=DEVICE
    )
)

test_metrics = evaluate(model, test_loader)
accuracy, precision, recall, f1, auc = test_metrics

print("\n===== FINAL TEST RESULTS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

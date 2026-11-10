import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix
)

# CONFIG
ROOT = Path(r"E:\DeepFakeX")
DATA_DIR = ROOT / "data" / "faces"
CHECKPOINT = ROOT / "best_efficientnet_b0_weighted.pth"

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# TRANSFORMS
weights = EfficientNet_B0_Weights.DEFAULT

test_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


# DATASET
class BinaryFaceDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for class_dir in self.root.iterdir():
            if not class_dir.is_dir():
                continue

            label = 0 if class_dir.name == "original" else 1

            for img_path in class_dir.rglob("*.jpg"):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# TEST DATA
test_dataset = BinaryFaceDataset(
    DATA_DIR / "test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# LOAD MODEL
model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

print("Weighted checkpoint loaded successfully!")


# EVALUATION
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(probabilities, dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(predictions.cpu().numpy())
        all_probs.extend(probabilities[:, 1].cpu().numpy())


# METRICS
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
auc = roc_auc_score(all_labels, all_probs)
balanced_acc = balanced_accuracy_score(all_labels, all_preds)

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

print("\n===== TEST CONFUSION MATRIX =====")
print(cm)

print("\n===== FINAL TEST RESULTS =====")
print(f"Accuracy              : {accuracy:.4f}")
print(f"Precision             : {precision:.4f}")
print(f"Recall                : {recall:.4f}")
print(f"F1 Score              : {f1:.4f}")
print(f"ROC-AUC               : {auc:.4f}")
print(f"Test Balanced Accuracy: {balanced_acc:.4f}")


# SAVE RESULTS
import numpy as np

np.savez(
    ROOT / "weighted_test_results.npz",
    labels=np.array(all_labels),
    predictions=np.array(all_preds),
    probabilities=np.array(all_probs),
    confusion_matrix=cm
)

print("\nResults saved to weighted_test_results.npz")

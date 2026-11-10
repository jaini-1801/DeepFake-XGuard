
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

ROOT = Path(r"E:\DeepFakeX")
DATA_DIR = ROOT / "data" / "faces"
CHECKPOINT = ROOT / "best_efficientnet_b0.pth"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4

print("Device:", DEVICE)


class BinaryFaceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        for folder in Path(root).iterdir():
            if not folder.is_dir():
                continue

            label = 0 if folder.name == "original" else 1

            for path in folder.rglob("*"):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


weights = EfficientNet_B0_Weights.DEFAULT
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

test_dataset = BinaryFaceDataset(
    DATA_DIR / "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2
)

state = torch.load(
    CHECKPOINT, map_location=DEVICE, weights_only=True
)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
auc = roc_auc_score(all_labels, all_probs)

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

print("\nConfusion Matrix (rows=true, columns=predicted):")
print("                 Predicted Real   Predicted Fake")
print(f"Actual Real          {cm[0, 0]:<12} {cm[0, 1]}")
print(f"Actual Fake          {cm[1, 0]:<12} {cm[1, 1]}")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real", "Fake"]
)
disp.plot()
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=300)
plt.close()

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

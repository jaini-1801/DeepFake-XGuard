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
# =========================

ROOT = Path(r"E:\DeepFakeX")
DATA_DIR = Path(r"E:\DeepFakeX\data\faces")

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# =========================
# TRANSFORMS
# =========================

weights = EfficientNet_B0_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

val_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])



class BinaryFaceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for class_dir in self.root.iterdir():
            if not class_dir.is_dir():
                continue

            # original = Real (0)
            # everything else = Fake (1)
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
# =========================
# DATASETS
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


print(
    f"Train: {len(train_dataset)}"
)

print(
    f"Validation: {len(val_dataset)}"
)

print(
    f"Test: {len(test_dataset)}"
)


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
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
# =========================

model = models.efficientnet_b0(
    weights=weights
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model = model.to(DEVICE)


# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# =========================
# EVALUATION FUNCTION
# =========================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs, dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_preds.extend(
                predictions.cpu().numpy()
            )

            all_probs.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

            print("Confusion Matrix:")
            print(cm)

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
    accuracy = accuracy_score(
             all_labels,
             all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probs
    )

    return accuracy, precision, recall, f1, auc




# =========================
# TRAINING
# =========================

best_f1 = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = (
        running_loss / len(train_loader)
    )

    metrics = evaluate(
        model,
        val_loader
    )

    accuracy, precision, recall, f1, auc = metrics

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Loss: {train_loss:.4f}"
    )

    print(
        f"Val Accuracy: {accuracy:.4f}"
    )

    print(
        f"Val Precision: {precision:.4f}"
    )

    print(
        f"Val Recall: {recall:.4f}"
    )

    print(
        f"Val F1: {f1:.4f}"
    )

    print(
        f"Val ROC-AUC: {auc:.4f}"
    )

    # Save best model based on F1
    if f1 > best_f1:

        best_f1 = f1

        torch.save(
            model.state_dict(),
            ROOT / "best_efficientnet_b0.pth"
        )

        print("✓ Best model saved")


# =========================
# FINAL TEST
# =========================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        ROOT / "best_efficientnet_b0.pth",
        map_location=DEVICE
    )
)

test_metrics = evaluate(
    model,
    test_loader
)

accuracy, precision, recall, f1, auc = test_metrics

print("\n===== FINAL TEST RESULTS =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")


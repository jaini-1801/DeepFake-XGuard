
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

ROOT = Path(r"E:\DeepFakeX")
DATA_DIR = ROOT / "data" / "faces"
RESULTS_DIR = ROOT / "results"
CHECKPOINT = ROOT / "best_efficientnet_b0_robust.pth"

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

weights = EfficientNet_B0_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std


class DegradationAugmentation:
    """Randomly apply JPEG compression, downscaling, or blur."""

    def __call__(self, image):
        choice = random.choice(["jpeg", "resize", "blur", "none"])

        if choice == "jpeg":
            quality = random.randint(30, 90)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")

        elif choice == "resize":
            factor = random.choice([0.25, 0.5, 0.75])
            w, h = image.size
            small = image.resize(
                (max(1, int(w * factor)), max(1, int(h * factor))),
                Image.Resampling.BILINEAR
            )
            image = small.resize((w, h), Image.Resampling.BILINEAR)

        elif choice == "blur":
            image = image.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0))
            )

        return image


train_transform = transforms.Compose([
    DegradationAugmentation(),
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


class BinaryFaceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for class_dir in self.root.iterdir():
            if not class_dir.is_dir():
                continue

            label = 0 if class_dir.name == "original" else 1

            self.samples.extend(
                (path, label) for path in class_dir.rglob("*.jpg")
            )

        if not self.samples:
            raise RuntimeError(f"No JPG images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = BinaryFaceDataset(DATA_DIR / "train", train_transform)
val_dataset = BinaryFaceDataset(DATA_DIR / "val", eval_transform)
test_dataset = BinaryFaceDataset(DATA_DIR / "test", eval_transform)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

model = models.efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=1e-4
)


def evaluate(loader):
    model.eval()
    labels_all, preds_all, probs_all = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            labels_all.extend(labels.numpy())
            preds_all.extend(probs.argmax(dim=1).cpu().numpy())
            probs_all.extend(probs[:, 1].cpu().numpy())

    return (
        accuracy_score(labels_all, preds_all),
        precision_score(labels_all, preds_all, zero_division=0),
        recall_score(labels_all, preds_all, zero_division=0),
        f1_score(labels_all, preds_all, zero_division=0),
        roc_auc_score(labels_all, probs_all),
    )


best_f1 = -1.0
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

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), CHECKPOINT)
        print("Saved robust checkpoint:", CHECKPOINT)

print("\nLoading best robust checkpoint...")
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))

accuracy, precision, recall, f1, auc = evaluate(test_loader)

report = (
    "Degradation-aware training — clean test set\n"
    f"Accuracy : {accuracy:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall   : {recall:.4f}\n"
    f"F1 Score : {f1:.4f}\n"
    f"ROC-AUC  : {auc:.4f}\n"
)

print("\n" + report)
(RESULTS_DIR / "robust_clean_metrics.txt").write_text(report)
print("Metrics saved to:", RESULTS_DIR / "robust_clean_metrics.txt")

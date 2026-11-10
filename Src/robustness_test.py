
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image, ImageFilter
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
import io
import csv

ROOT = Path(r"E:\DeepFakeX")
DATA_DIR = ROOT / "data" / "faces" / "test"
CHECKPOINT = ROOT / "best_efficientnet_b0.pth"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4

weights = EfficientNet_B0_Weights.DEFAULT
normalize = transforms.Normalize(
    mean=weights.transforms().mean,
    std=weights.transforms().std
)
to_tensor = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    normalize
])

# Severity settings for each degradation.
def degrade(image, mode, severity):
    if mode == "clean":
        return image
    if mode == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=severity)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    if mode == "resize":
        w, h = image.size
        small = image.resize(
            (max(1, int(w * severity)), max(1, int(h * severity))),
            Image.Resampling.BILINEAR
        )
        return small.resize((w, h), Image.Resampling.BILINEAR)
    if mode == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=severity))
    return image


class TestImages(torch.utils.data.Dataset):
    def __init__(self, root):
        self.samples = []
        for folder in Path(root).iterdir():
            if folder.is_dir():
                label = 0 if folder.name == "original" else 1
                for path in folder.rglob("*"):
                    if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
        return image, label


dataset = TestImages(DATA_DIR)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(
    CHECKPOINT, map_location=DEVICE, weights_only=True
))
model.to(DEVICE).eval()

conditions = [
    ("clean", 0),
    ("jpeg", 90), ("jpeg", 70), ("jpeg", 50), ("jpeg", 30),
    ("resize", 0.75), ("resize", 0.50), ("resize", 0.25),
    ("blur", 1), ("blur", 2), ("blur", 3), ("blur", 4),
]

results = []

for mode, severity in conditions:
    labels_all, preds_all, probs_all = [], [], []

    for start in range(0, len(dataset), BATCH_SIZE):
        batch = dataset.samples[start:start + BATCH_SIZE]
        images, labels = [], []

        for path, label in batch:
            with Image.open(path) as im:
                image = im.convert("RGB")
            image = degrade(image, mode, severity)
            images.append(to_tensor(image))
            labels.append(label)

        x = torch.stack(images).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()

        labels_all.extend(labels)
        preds_all.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())

    f1 = f1_score(labels_all, preds_all, zero_division=0)
    auc = roc_auc_score(labels_all, probs_all)

    results.append({
        "condition": mode,
        "severity": severity,
        "f1": f1,
        "roc_auc": auc
    })

    print(f"{mode:6} severity={severity}: F1={f1:.4f}, AUC={auc:.4f}")

with open(RESULTS / "robustness_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["condition", "severity", "f1", "roc_auc"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", RESULTS / "robustness_results.csv")

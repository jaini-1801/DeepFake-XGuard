"""
robustness_test.py
------------------
Tests how well the trained model holds up under real-world image degradations.

Motivation:
  Real deepfake videos shared online are often compressed (WhatsApp, YouTube),
  blurry, or low-resolution. A model that only works on clean images is not
  practically useful.

What this script does:
  For each degradation type and severity (defined in 'conditions' list),
  it applies that degradation to every test image and measures F1 + AUC.

Degradation types:
  - clean:  no changes (baseline)
  - jpeg:   JPEG compression at various qualities (90, 70, 50, 30)
             lower quality = more blocky artifacts
  - resize: downscale then upscale to 75%, 50%, or 25%
             simulates pixelated / low-res video
  - blur:   Gaussian blur with radius 1-4
             simulates camera blur or motion blur

Output:
  results/robustness_results.csv  -- F1 and AUC for every condition
"""

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

# ---- Config ----
ROOT       = Path(r"E:\DeepFakeX")
DATA_DIR   = ROOT / "data" / "faces" / "test"   # Test face images
CHECKPOINT = ROOT / "best_efficientnet_b0.pth"  # Baseline model (no weighting/robust training)
RESULTS    = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4

# ---- Image normalization (same as training) ----
weights   = EfficientNet_B0_Weights.DEFAULT
normalize = transforms.Normalize(
    mean=weights.transforms().mean,
    std=weights.transforms().std
)
# Standard preprocessing: resize to 160x160, convert to tensor, normalize
to_tensor = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    normalize
])


def degrade(image, mode, severity):
    """
    Apply a specific degradation to a PIL image.

    Args:
        image:    PIL Image to degrade
        mode:     type of degradation ('clean', 'jpeg', 'resize', 'blur')
        severity: degree of degradation (meaning depends on mode):
                  - jpeg:   quality value (lower = worse, e.g. 30)
                  - resize: scale factor (smaller = worse, e.g. 0.25)
                  - blur:   blur radius (larger = blurrier, e.g. 4)
                  - clean:  ignored (no change)

    Returns: degraded PIL Image
    """
    if mode == "clean":
        return image  # No change

    if mode == "jpeg":
        # Re-save as JPEG at the given quality, then reload
        # Lower quality = stronger compression artifacts
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=severity)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()

    if mode == "resize":
        # Scale down to (severity * original size), then back up
        # e.g. severity=0.25 -> shrink to 25%, then enlarge back
        w, h = image.size
        small = image.resize(
            (max(1, int(w * severity)), max(1, int(h * severity))),
            Image.Resampling.BILINEAR
        )
        return small.resize((w, h), Image.Resampling.BILINEAR)

    if mode == "blur":
        # Apply Gaussian blur with the given radius
        return image.filter(ImageFilter.GaussianBlur(radius=severity))

    return image  # Fallback: no change


class TestImages(torch.utils.data.Dataset):
    """
    Simple dataset that loads all test face images without any transforms.
    Transforms are applied manually per-batch after degradation is applied.

    Label: 0 = Real (from 'original' folder), 1 = Fake (all other folders)
    """

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
        # Return raw PIL image (degradation applied later)
        return image, label


# ---- Load test dataset ----
dataset = TestImages(DATA_DIR)

# ---- Load model ----
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
)
model.to(DEVICE).eval()

# ---- Define all conditions to test ----
# Each entry is (mode, severity) -- severity meaning depends on mode
conditions = [
    ("clean",  0),                                        # Baseline: no degradation
    ("jpeg",  90), ("jpeg",  70), ("jpeg",  50), ("jpeg", 30),  # JPEG quality levels
    ("resize", 0.75), ("resize", 0.50), ("resize", 0.25),       # Downscale ratios
    ("blur",   1),  ("blur",    2),  ("blur",    3),  ("blur", 4), # Blur radii
]

results = []  # Will store one row per condition

for mode, severity in conditions:

    labels_all = []
    preds_all  = []
    probs_all  = []

    # Manual batching: iterate in chunks of BATCH_SIZE
    for start in range(0, len(dataset), BATCH_SIZE):
        batch = dataset.samples[start:start + BATCH_SIZE]
        images = []
        labels = []

        for path, label in batch:
            with Image.open(path) as im:
                image = im.convert("RGB")
            # Apply degradation to the raw image before preprocessing
            image = degrade(image, mode, severity)
            # Then apply standard resize + normalize
            images.append(to_tensor(image))
            labels.append(label)

        # Stack list of tensors into one batch tensor
        x = torch.stack(images).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[:, 1]  # Fake probability
            preds  = (probs >= 0.5).long()                # Threshold at 0.5

        labels_all.extend(labels)
        preds_all.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())

    # Compute metrics for this condition
    f1  = f1_score(labels_all, preds_all, zero_division=0)
    auc = roc_auc_score(labels_all, probs_all)

    results.append({
        "condition": mode,
        "severity":  severity,
        "f1":        f1,
        "roc_auc":   auc
    })

    print(f"{mode:6} severity={severity}: F1={f1:.4f}, AUC={auc:.4f}")

# ---- Save all results to CSV ----
with open(RESULTS / "robustness_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["condition", "severity", "f1", "roc_auc"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", RESULTS / "robustness_results.csv")

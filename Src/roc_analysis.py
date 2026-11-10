
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(r"E:\DeepFakeX")

CSV_PATH = ROOT / "results/celebdf/celebdf_video_predictions.csv"
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load predictions
df = pd.read_csv(CSV_PATH)

y_true = df["true_label"].astype(int)
y_score = df["fake_probability"].astype(float)

# Threshold-based predictions
y_pred = (y_score >= 0.5).astype(int)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2,
         label=f"EfficientNet-B0 (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--",
         label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Celeb-DF Video-Level Evaluation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "celebdf_roc_curve.png", dpi=300)
plt.show()

# Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(
    y_true, y_score
)

ap = average_precision_score(y_true, y_score)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2,
         label=f"EfficientNet-B0 (AP = {ap:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Celeb-DF")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "celebdf_precision_recall.png", dpi=300)
plt.show()

# Classification report
print("\nROC-AUC:", round(auc, 4))
print("Average Precision:", round(ap, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=["Real", "Fake"]
))

print("\nGraphs saved in:", OUTPUT)

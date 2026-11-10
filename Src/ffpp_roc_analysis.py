
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

ROOT = Path(r"E:\DeepFakeX")
DATA = np.load(ROOT / "weighted_test_results.npz")

y_true = DATA["labels"]
y_score = DATA["probabilities"]

fpr, tpr, _ = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"EfficientNet-B0 (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], "--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - FaceForensics++ Test Set")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "figures" / "ffpp_roc_curve.png", dpi=300)
plt.show()

precision, recall, _ = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"EfficientNet-B0 (AP = {ap:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - FaceForensics++")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "figures" / "ffpp_precision_recall.png", dpi=300)
plt.show()

print(f"FF++ ROC-AUC: {auc:.4f}")
print(f"FF++ Average Precision: {ap:.4f}")

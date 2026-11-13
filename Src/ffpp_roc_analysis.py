"""
ffpp_roc_analysis.py
--------------------
Generates ROC curve and Precision-Recall curve for the
FaceForensics++ (FF++) test set.

Reads the saved predictions from train_evaluate.py
(stored in weighted_test_results.npz) and plots two curves:

  1. ROC Curve: shows the tradeoff between catching fakes (TPR)
     vs falsely flagging real videos (FPR) at every threshold.
     AUC = 1.0 means perfect, AUC = 0.5 means random guessing.

  2. Precision-Recall Curve: shows how precision drops as recall increases.
     AP (Average Precision) = area under this curve.

Output:
  figures/ffpp_roc_curve.png
  figures/ffpp_precision_recall.png
"""

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

# Load saved predictions (created by train_evaluate.py)
# .npz is a compressed numpy archive: contains 'labels' and 'probabilities' arrays
DATA = np.load(ROOT / "weighted_test_results.npz")

y_true  = DATA["labels"]        # Ground truth: 0=Real, 1=Fake
y_score = DATA["probabilities"] # Model's predicted Fake probability


# ==========================================
# ROC CURVE
# ==========================================

# Compute (FPR, TPR) pairs at every threshold the model assigns
fpr, tpr, _ = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

plt.figure(figsize=(8, 6))

# Model's ROC curve -- the higher and to the left, the better
plt.plot(fpr, tpr, label=f"EfficientNet-B0 (AUC = {auc:.4f})")

# Diagonal line = random classifier (no skill)
plt.plot([0, 1], [0, 1], "--", label="Random Classifier")

plt.xlabel("False Positive Rate")   # Rate of real videos called fake
plt.ylabel("True Positive Rate")    # Rate of fake videos correctly caught
plt.title("ROC Curve - FaceForensics++ Test Set")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "figures" / "ffpp_roc_curve.png", dpi=300)
plt.show()


# ==========================================
# PRECISION-RECALL CURVE
# ==========================================

# Compute (precision, recall) pairs at every threshold
precision, recall, _ = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)  # Area under PR curve

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"EfficientNet-B0 (AP = {ap:.4f})")

plt.xlabel("Recall")       # Of all fakes, how many did we catch
plt.ylabel("Precision")    # Of flagged items, how many are truly fake
plt.title("Precision-Recall Curve - FaceForensics++")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "figures" / "ffpp_precision_recall.png", dpi=300)
plt.show()


# ---- Print summary ----
print(f"FF++ ROC-AUC: {auc:.4f}")
print(f"FF++ Average Precision: {ap:.4f}")

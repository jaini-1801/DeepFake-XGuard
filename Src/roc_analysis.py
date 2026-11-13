"""
roc_analysis.py
---------------
Generates ROC curve and Precision-Recall curve plots for the
Celeb-DF video-level predictions.

What is a ROC curve?
  - X-axis: False Positive Rate (FPR) = how often we falsely call real videos fake
  - Y-axis: True Positive Rate (TPR/Recall) = how often we correctly detect fakes
  - AUC (Area Under Curve): 1.0 = perfect, 0.5 = random guessing

What is a Precision-Recall curve?
  - X-axis: Recall = how many fakes we caught
  - Y-axis: Precision = of the ones we flagged, how many were actually fake
  - AP (Average Precision): summary score, 1.0 = perfect

These curves visualize performance at all possible thresholds, not just 0.5.

Input:
  results/celebdf/celebdf_video_predictions.csv  (from evaluate_celebdf.py)

Output:
  figures/celebdf_roc_curve.png
  figures/celebdf_precision_recall.png
"""

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

# ---- Config ----
ROOT = Path(r"E:\DeepFakeX")

# Input: video-level predictions from evaluate_celebdf.py
CSV_PATH = ROOT / "results/celebdf/celebdf_video_predictions.csv"

# Output folder for plots
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---- Load predictions ----
df = pd.read_csv(CSV_PATH)

y_true  = df["true_label"].astype(int)       # Ground truth: 0=Real, 1=Fake
y_score = df["fake_probability"].astype(float) # Model's predicted Fake probability

# Apply a fixed threshold of 0.5 for the classification report
y_pred = (y_score >= 0.5).astype(int)


# ==========================================
# ROC CURVE
# ==========================================

# roc_curve returns arrays of (FPR, TPR) for every possible threshold
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

plt.figure(figsize=(8, 6))

# Plot the model's ROC curve
plt.plot(fpr, tpr, linewidth=2,
         label=f"EfficientNet-B0 (AUC = {auc:.4f})")

# Plot the diagonal: a random classifier would follow this line
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")   # How often we incorrectly say "Fake"
plt.ylabel("True Positive Rate")    # How often we correctly detect "Fake"
plt.title("ROC Curve - Celeb-DF Video-Level Evaluation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "celebdf_roc_curve.png", dpi=300)
plt.show()


# ==========================================
# PRECISION-RECALL CURVE
# ==========================================

# precision_recall_curve returns (precision, recall) arrays at every threshold
precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

# AP = area under the PR curve (weighted average of precision at each recall level)
ap = average_precision_score(y_true, y_score)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2,
         label=f"EfficientNet-B0 (AP = {ap:.4f})")

plt.xlabel("Recall")       # How many actual fakes we caught
plt.ylabel("Precision")    # Of flagged videos, how many were actually fake
plt.title("Precision-Recall Curve - Celeb-DF")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "celebdf_precision_recall.png", dpi=300)
plt.show()


# ==========================================
# SUMMARY METRICS (at threshold = 0.5)
# ==========================================

print("\nROC-AUC:", round(auc, 4))
print("Average Precision:", round(ap, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
# Rows = actual class, Columns = predicted class
# [TN FP]   <- actual Real
# [FN TP]   <- actual Fake

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=["Real", "Fake"]
))

print("\nGraphs saved in:", OUTPUT)

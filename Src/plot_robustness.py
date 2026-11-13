"""
plot_robustness.py
------------------
Visualizes the robustness results from robustness_test.py.

Reads the CSV file produced by robustness_test.py and creates a 3x2 grid
of line plots showing how the model's F1 score and ROC-AUC change as
image quality degrades.

For each degradation type (JPEG, Resize, Blur):
  - Left plot: F1 score vs degradation level
  - Right plot: ROC-AUC vs degradation level
  - Dashed line: clean baseline (no degradation) for comparison

Output:
  figures/robustness_analysis.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"E:\DeepFakeX")

# Load the robustness results CSV (produced by robustness_test.py)
df = pd.read_csv(ROOT / "results" / "robustness_results.csv")

# Hardcoded clean baseline scores (from evaluate.py on the unmodified test set)
baseline_f1  = 0.9302325581
baseline_auc = 0.9006943878

# The three degradation types to plot
conditions = ["jpeg", "resize", "blur"]

# ---- Figure setup ----
# 3 rows (one per degradation type) x 2 columns (F1 and AUC)
fig, axes = plt.subplots(
    3, 2,
    figsize=(14, 13),
    constrained_layout=True  # Automatically adjust spacing to avoid overlap
)

fig.suptitle(
    "EfficientNet-B0: Robustness Analysis",
    fontsize=19,
    fontweight="bold"
)

# Human-readable names for the plot titles
titles = {
    "jpeg":   "JPEG Compression",
    "resize": "Image Resizing",
    "blur":   "Gaussian Blur"
}

for i, condition in enumerate(conditions):

    # Get rows for this degradation type only
    subset = df[df["condition"] == condition].copy()
    subset["severity"] = pd.to_numeric(subset["severity"])
    subset = subset.sort_values("severity")  # Order by increasing degradation

    # Create readable x-axis labels for each degradation type
    if condition == "jpeg":
        # e.g. Q90, Q70, Q50, Q30 (JPEG quality values)
        labels = [f"Q{int(x)}" for x in subset["severity"]]
    elif condition == "resize":
        # e.g. 75%, 50%, 25% (downscale percentage)
        labels = [f"{int(x * 100)}%" for x in subset["severity"]]
    else:
        # e.g. Level 1, Level 2, Level 3, Level 4 (blur radius)
        labels = [f"Level {int(x)}" for x in subset["severity"]]

    # ---- Left plot: F1 score ----
    ax1 = axes[i, 0]
    ax1.plot(
        labels, subset["f1"],
        marker="o", markersize=7, linewidth=2.5,
        label="Degraded images"
    )
    # Dashed line showing what F1 was on clean images
    ax1.axhline(
        baseline_f1, linestyle="--", linewidth=2,
        label=f"Clean baseline ({baseline_f1:.3f})"
    )
    ax1.set_title(f"{titles[condition]} - F1-score", fontsize=13, pad=12)
    ax1.set_xlabel("Degradation level")
    ax1.set_ylabel("F1-score")
    ax1.set_ylim(0.45, 1.02)   # Fixed y-axis range for easy comparison between plots
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, loc="lower left")

    # ---- Right plot: ROC-AUC ----
    ax2 = axes[i, 1]
    ax2.plot(
        labels, subset["roc_auc"],
        marker="o", markersize=7, linewidth=2.5,
        label="Degraded images"
    )
    # Dashed line showing clean baseline AUC
    ax2.axhline(
        baseline_auc, linestyle="--", linewidth=2,
        label=f"Clean baseline ({baseline_auc:.3f})"
    )
    ax2.set_title(f"{titles[condition]} - ROC-AUC", fontsize=13, pad=12)
    ax2.set_xlabel("Degradation level")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_ylim(0.45, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, loc="lower left")

# ---- Save the figure ----
output = ROOT / "figures" / "robustness_analysis.png"

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print(f"Graph saved successfully: {output}")

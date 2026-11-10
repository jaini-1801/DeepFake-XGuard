
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"E:\DeepFakeX")
df = pd.read_csv(ROOT / "results" / "robustness_results.csv")

baseline_f1 = 0.9302325581
baseline_auc = 0.9006943878

conditions = ["jpeg", "resize", "blur"]

# Larger figure with automatic spacing
fig, axes = plt.subplots(
    3, 2,
    figsize=(14, 13),
    constrained_layout=True
)

fig.suptitle(
    "EfficientNet-B0: Robustness Analysis",
    fontsize=19,
    fontweight="bold"
)

titles = {
    "jpeg": "JPEG Compression",
    "resize": "Image Resizing",
    "blur": "Gaussian Blur"
}

for i, condition in enumerate(conditions):

    subset = df[df["condition"] == condition].copy()
    subset["severity"] = pd.to_numeric(subset["severity"])
    subset = subset.sort_values("severity")

    # Meaningful x-axis labels
    if condition == "jpeg":
        labels = [f"Q{int(x)}" for x in subset["severity"]]
    elif condition == "resize":
        labels = [f"{int(x * 100)}%" for x in subset["severity"]]
    else:
        labels = [f"Level {int(x)}" for x in subset["severity"]]

    # F1-score
    ax1 = axes[i, 0]
    ax1.plot(
        labels, subset["f1"],
        marker="o", markersize=7, linewidth=2.5,
        label="Degraded images"
    )
    ax1.axhline(
        baseline_f1, linestyle="--", linewidth=2,
        label=f"Clean baseline ({baseline_f1:.3f})"
    )

    ax1.set_title(
        f"{titles[condition]} - F1-score",
        fontsize=13, pad=12
    )
    ax1.set_xlabel("Degradation level")
    ax1.set_ylabel("F1-score")
    ax1.set_ylim(0.45, 1.02)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, loc="lower left")

    # ROC-AUC
    ax2 = axes[i, 1]
    ax2.plot(
        labels, subset["roc_auc"],
        marker="o", markersize=7, linewidth=2.5,
        label="Degraded images"
    )
    ax2.axhline(
        baseline_auc, linestyle="--", linewidth=2,
        label=f"Clean baseline ({baseline_auc:.3f})"
    )

    ax2.set_title(
        f"{titles[condition]} - ROC-AUC",
        fontsize=13, pad=12
    )
    ax2.set_xlabel("Degradation level")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_ylim(0.45, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, loc="lower left")

output = ROOT / "figures" / "robustness_analysis.png"

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print(f"Graph saved successfully: {output}")

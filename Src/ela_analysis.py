"""
ela_analysis.py
---------------
Error Level Analysis (ELA) for visualizing deepfake artifacts.

What is ELA?
  ELA reveals how much each part of an image was modified by JPEG compression.
  Genuine (untouched) images have uniform compression across the whole image.
  Manipulated/fake images show HIGHER error in regions that were edited,
  because those regions have different compression history.

How it works:
  1. Take the original image
  2. Re-save it as JPEG at a fixed quality (e.g. 90)
  3. Compute the absolute pixel difference (original vs re-saved)
  4. Bright spots in the difference map = heavily modified regions

This script:
  - Computes ELA maps for FF++ and Celeb-DF (real + fake faces)
  - Generates 3 figures:
      Figure 1: Qualitative examples (image + its ELA map, side by side)
      Figure 2: Average ELA heatmap per class
      Figure 3: Distribution of ELA intensity per class (violin plot)
  - Runs a Mann-Whitney U test to check if real vs fake are statistically different
"""

from pathlib import Path
import io, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy import stats

# ---- Config ----
ROOT = Path(r"E:\DeepFakeX")
OUT  = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Which groups to analyze (name -> folder path)
GROUPS = {
    "FF++ Real":    ROOT / "Data/faces/test/original",
    "FF++ Fake":    ROOT / "Data/faces/test/fake",          # Adjust if path differs
    "CelebDF Real": ROOT / "Data/faces/celebdf_test/Celeb-real",
    "CelebDF Fake": ROOT / "Data/faces/celebdf_test/Celeb-synthesis",
}

QUALITY  = 90    # JPEG quality used for re-compression (lower = more compression)
SIZE     = 224   # Resize all images to 224x224 for consistent comparison
N_STATS  = 300   # How many images per group to use for statistics
N_SHOW   = 3     # How many example images to show in Figure 1
SEED     = 42    # Random seed for reproducibility

random.seed(SEED)
np.random.seed(SEED)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


# ---- Helper: load image ----
def load(path):
    """Load an image, convert to RGB, and resize to SIZExSIZE."""
    return Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)


# ---- ELA computation ----
def ela_map(img, quality=QUALITY):
    """
    Compute the ELA (Error Level Analysis) map for an image.

    Steps:
      1. Re-save the image as JPEG at given quality (introduces compression)
      2. Reload the re-saved image
      3. Compute absolute pixel difference: |original - recompressed|
      4. Average across RGB channels -> one value per pixel

    Important: maps are NOT normalized per image, so values are comparable
    across images (bright = more error = more likely manipulated).
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)   # Re-compress
    buf.seek(0)
    rec  = Image.open(buf).convert("RGB")            # Reload re-compressed image
    diff = np.abs(
        np.asarray(img,  np.float32) -
        np.asarray(rec, np.float32)
    )
    return diff.mean(axis=2)  # Average over R, G, B channels


def list_images(folder):
    """Return all .jpg files in a folder, or [] if folder doesn't exist."""
    if not folder.exists():
        print("Skipping (not found):", folder)
        return []
    return sorted(folder.rglob("*.jpg"))


# ---- Collect ELA data ----
# For each group: randomly sample up to N_STATS images and compute their ELA maps
data = {}   # group name -> {"paths": [...], "maps": numpy array}

for name, folder in GROUPS.items():
    files = list_images(folder)
    if not files:
        continue
    # Randomly pick up to N_STATS images to avoid very long compute times
    chosen = random.sample(files, min(N_STATS, len(files)))
    # Stack all ELA maps into one array: shape (N, H, W)
    maps = np.stack([ela_map(load(p)) for p in chosen])
    data[name] = {"paths": chosen, "maps": maps}
    print(f"{name}: {len(chosen)} images, mean ELA = {maps.mean():.3f}")

names = list(data)

# Shared color scale: use the 99.5th percentile of all pixels across all groups
# This prevents one extreme outlier from washing out the colormap
VMAX = float(np.percentile(
    np.concatenate([d["maps"].ravel() for d in data.values()]), 99.5
))


# ==========================================
# FIGURE 1: Qualitative examples
# Shows N_SHOW examples per group: original image + its ELA map side-by-side
# ==========================================

fig = plt.figure(figsize=(3.0 * 2 * N_SHOW + 1, 3.1 * len(names)))
gs  = GridSpec(
    len(names), 2 * N_SHOW + 1,
    width_ratios=[1] * (2 * N_SHOW) + [0.05],  # Last column = colorbar
    wspace=0.05, hspace=0.25
)
im = None

for r, name in enumerate(names):
    for c in range(N_SHOW):
        p = data[name]["paths"][c]
        m = data[name]["maps"][c]

        # Left: original image
        ax = fig.add_subplot(gs[r, 2 * c])
        ax.imshow(load(p))
        ax.axis("off")
        if r == 0:
            ax.set_title("Input", fontsize=10)
        if c == 0:
            ax.text(-0.08, 0.5, name, transform=ax.transAxes, rotation=90,
                    va="center", ha="right", fontsize=12, fontweight="bold")

        # Right: ELA map (brighter = more error = more likely manipulated)
        ax = fig.add_subplot(gs[r, 2 * c + 1])
        im = ax.imshow(m, cmap="inferno", vmin=0, vmax=VMAX)
        ax.axis("off")
        if r == 0:
            ax.set_title("ELA", fontsize=10)

# Add shared colorbar on the right
cax = fig.add_subplot(gs[:, -1])
fig.colorbar(im, cax=cax).set_label(f"Mean |error| (0-255), JPEG q={QUALITY}")

fig.suptitle(
    "Error Level Analysis: randomly sampled examples (shared colour scale)",
    fontsize=14, y=0.995
)
fig.savefig(OUT / "ela_qualitative.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "ela_qualitative.pdf", bbox_inches="tight")


# ==========================================
# FIGURE 2: Average ELA heatmap per class
# Shows the mean ELA map across all sampled images in each group
# Reveals which face regions are most often manipulated
# ==========================================

fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names) + 1, 3.6))
axes = np.atleast_1d(axes)

for ax, name in zip(axes, names):
    # Average all individual ELA maps for this group
    mean_map = data[name]["maps"].mean(axis=0)
    im = ax.imshow(mean_map, cmap="inferno", vmin=0, vmax=VMAX * 0.6)
    ax.set_title(f"{name}\n(n={len(data[name]['maps'])})", fontsize=11)
    ax.axis("off")

fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label(
    "Mean ELA over images"
)
fig.suptitle("Average ELA map per class", fontsize=14)
fig.savefig(OUT / "ela_mean_maps.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "ela_mean_maps.pdf", bbox_inches="tight")


# ==========================================
# FIGURE 3: Distribution of ELA intensity
# Violin plot showing the spread of per-image mean ELA for each class
# If real and fake distributions are separated, ELA is a useful signal
# ==========================================

# Per-image mean ELA: average all pixel values in each image's ELA map
scores = {n: d["maps"].mean(axis=(1, 2)) for n, d in data.items()}

fig, ax = plt.subplots(figsize=(7.5, 4.5))
parts = ax.violinplot(
    [scores[n] for n in names],
    showmedians=True,
    showextrema=False
)
for pc in parts["bodies"]:
    pc.set_alpha(0.6)

ax.set_xticks(range(1, len(names) + 1))
ax.set_xticklabels(names, rotation=15)
ax.set_ylabel("Per-image mean ELA")
ax.set_title("Distribution of ELA intensity")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "ela_distribution.png", dpi=300)
fig.savefig(OUT / "ela_distribution.pdf")


# ==========================================
# STATISTICAL COMPARISON
# Mann-Whitney U test: checks if real and fake ELA distributions are
# significantly different (p < 0.05 means they are statistically different)
# AUC = 0.5 means no separation; AUC closer to 0 or 1 = strong separation
# ==========================================

def compare(real, fake):
    """Run a statistical test comparing ELA between a real and fake group."""
    if real in scores and fake in scores:
        a, b = scores[real], scores[fake]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc  = u / (len(a) * len(b))   # P(real ELA > fake ELA); 0.5 = no separation
        print(
            f"{real} vs {fake}: "
            f"median {np.median(a):.3f} vs {np.median(b):.3f}, "
            f"Mann-Whitney p={p:.2e}, AUC={auc:.3f}"
        )

print("\n--- Statistical comparison ---")
compare("CelebDF Real", "CelebDF Fake")
compare("FF++ Real",    "FF++ Fake")

plt.show()

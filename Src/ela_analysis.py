from pathlib import Path
import io, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy import stats

# ---------------- CONFIG ----------------
ROOT = Path(r"E:\DeepFakeX")
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "FF++ Real":    ROOT / "Data/faces/test/original",
    # adjust this path to your FF++ fake folder (skipped if missing)
    "FF++ Fake":    ROOT / "Data/faces/test/fake",
    "CelebDF Real": ROOT / "Data/faces/celebdf_test/Celeb-real",
    "CelebDF Fake": ROOT / "Data/faces/celebdf_test/Celeb-synthesis",
}

QUALITY = 90        # JPEG re-compression quality
SIZE = 224          # common resolution for all images
N_STATS = 300       # images per group for aggregate analysis
N_SHOW = 3          # examples per group in the qualitative figure
SEED = 42
random.seed(SEED); np.random.seed(SEED)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

# ---------------- ELA ----------------
def load(path):
    return Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)

def ela_map(img, quality=QUALITY):
    """Absolute difference after JPEG recompression, averaged over RGB.
    NOT normalised per image, so maps are comparable across images."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    rec = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(img, np.float32) - np.asarray(rec, np.float32))
    return diff.mean(axis=2)

def list_images(folder):
    if not folder.exists():
        print("Skipping (not found):", folder)
        return []
    return sorted(folder.rglob("*.jpg"))

# ---------------- COLLECT DATA ----------------
data = {}   # name -> dict(paths, maps)
for name, folder in GROUPS.items():
    files = list_images(folder)
    if not files:
        continue
    chosen = random.sample(files, min(N_STATS, len(files)))
    maps = np.stack([ela_map(load(p)) for p in chosen])
    data[name] = {"paths": chosen, "maps": maps}
    print(f"{name}: {len(chosen)} images, mean ELA = {maps.mean():.3f}")

names = list(data)
# One shared colour scale for every panel (99.5th percentile of all pixels)
VMAX = float(np.percentile(np.concatenate([d["maps"].ravel() for d in data.values()]), 99.5))

# ---------------- FIGURE 1: qualitative examples ----------------
fig = plt.figure(figsize=(3.0 * 2 * N_SHOW + 1, 3.1 * len(names)))
gs = GridSpec(len(names), 2 * N_SHOW + 1, width_ratios=[1] * (2 * N_SHOW) + [0.05],
              wspace=0.05, hspace=0.25)
im = None
for r, name in enumerate(names):
    for c in range(N_SHOW):
        p, m = data[name]["paths"][c], data[name]["maps"][c]
        ax = fig.add_subplot(gs[r, 2 * c]);     ax.imshow(load(p)); ax.axis("off")
        if r == 0: ax.set_title("Input", fontsize=10)
        if c == 0: ax.text(-0.08, 0.5, name, transform=ax.transAxes, rotation=90,
                           va="center", ha="right", fontsize=12, fontweight="bold")
        ax = fig.add_subplot(gs[r, 2 * c + 1])
        im = ax.imshow(m, cmap="inferno", vmin=0, vmax=VMAX); ax.axis("off")
        if r == 0: ax.set_title("ELA", fontsize=10)
cax = fig.add_subplot(gs[:, -1])
fig.colorbar(im, cax=cax).set_label(f"Mean |error| (0-255), JPEG q={QUALITY}")
fig.suptitle("Error Level Analysis: randomly sampled examples (shared colour scale)",
             fontsize=14, y=0.995)
fig.savefig(OUT / "ela_qualitative.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "ela_qualitative.pdf", bbox_inches="tight")

# ---------------- FIGURE 2: mean ELA heatmap per class ----------------
fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names) + 1, 3.6))
axes = np.atleast_1d(axes)
for ax, name in zip(axes, names):
    mean_map = data[name]["maps"].mean(axis=0)
    im = ax.imshow(mean_map, cmap="inferno", vmin=0, vmax=VMAX * 0.6)
    ax.set_title(f"{name}\n(n={len(data[name]['maps'])})", fontsize=11)
    ax.axis("off")
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label("Mean ELA over images")
fig.suptitle("Average ELA map per class", fontsize=14)
fig.savefig(OUT / "ela_mean_maps.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "ela_mean_maps.pdf", bbox_inches="tight")

# ---------------- FIGURE 3: distributions + statistics ----------------
scores = {n: d["maps"].mean(axis=(1, 2)) for n, d in data.items()}   # per-image mean ELA

fig, ax = plt.subplots(figsize=(7.5, 4.5))
parts = ax.violinplot([scores[n] for n in names], showmedians=True, showextrema=False)
for pc in parts["bodies"]:
    pc.set_alpha(0.6)
ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, rotation=15)
ax.set_ylabel("Per-image mean ELA")
ax.set_title("Distribution of ELA intensity")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "ela_distribution.png", dpi=300)
fig.savefig(OUT / "ela_distribution.pdf")

def compare(real, fake):
    if real in scores and fake in scores:
        a, b = scores[real], scores[fake]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = u / (len(a) * len(b))        # P(real > fake); 0.5 = no separation
        print(f"{real} vs {fake}: median {np.median(a):.3f} vs {np.median(b):.3f}, "
              f"Mann-Whitney p={p:.2e}, AUC={auc:.3f}")

print("\n--- Statistical comparison ---")
compare("CelebDF Real", "CelebDF Fake")
compare("FF++ Real", "FF++ Fake")
plt.show()
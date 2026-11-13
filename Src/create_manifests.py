"""
create_manifests.py
-------------------
Generates CSV manifest files (train.csv, val.csv, test.csv) for the
FaceForensics++ dataset.

A "manifest" is just a list of all video files for a given split,
with their path, dataset name, video ID, and label (0=Real, 1=Fake).

The official train/val/test split comes from JSON files provided by FF++.
Each JSON file contains pairs of video IDs like ["001", "002"] meaning
video 001 and video 002 were swapped to create a deepfake pair.

Output CSV columns:
  - video_path: full path to the .mp4 file
  - dataset:    e.g. 'original', 'Deepfakes', 'Face2Face'
  - video_id:   e.g. '001' or '001_002' for manipulated pairs
  - label:      0 (Real) or 1 (Fake)
"""

import json
import csv
from pathlib import Path

# ---- Paths ----
ROOT   = Path(r"E:\DeepFakeX")
DATA   = ROOT / "data"
SPLITS = DATA / "splits"    # Contains train.json, val.json, test.json from FF++
OUTPUT = DATA / "manifests" # Where the CSV files will be saved

OUTPUT.mkdir(parents=True, exist_ok=True)

# ---- Dataset definitions ----
# Maps each dataset name to its video folder and label (0=Real, 1=Fake)
DATASETS = {
    "original": {
        "path": DATA / "original_sequences" / "youtube" / "c40" / "videos",
        "label": 0   # Real faces
    },
    "Deepfakes": {
        "path": DATA / "manipulated_sequences" / "Deepfakes" / "c40" / "videos",
        "label": 1   # Fake: face swap using deep learning
    },
    "Face2Face": {
        "path": DATA / "manipulated_sequences" / "Face2Face" / "c40" / "videos",
        "label": 1   # Fake: facial reenactment
    },
    "FaceShifter": {
        "path": DATA / "manipulated_sequences" / "FaceShifter" / "c40" / "videos",
        "label": 1   # Fake: high-fidelity face swap
    },
    "FaceSwap": {
        "path": DATA / "manipulated_sequences" / "FaceSwap" / "c40" / "videos",
        "label": 1   # Fake: graphical face swap
    },
    "NeuralTextures": {
        "path": DATA / "manipulated_sequences" / "NeuralTextures" / "c40" / "videos",
        "label": 1   # Fake: neural texture-based manipulation
    }
}


def load_split(name):
    """Load the official FF++ split JSON file for a given split (train/val/test)."""
    with open(SPLITS / f"{name}.json", "r") as f:
        return json.load(f)


def process_split(split_name):
    """
    Build the manifest CSV for one split (train, val, or test).

    The JSON contains video ID pairs like [["001", "002"], ["003", "004"]] --
    meaning video 001 was used to create a deepfake of 002, and vice versa.
    We expand these bidirectionally so both directions are included.
    """

    split_data = load_split(split_name)

    # Build a set of all source-target pairs (both directions)
    # e.g. ["001", "002"] -> (001, 002) and (002, 001)
    pairs = set()
    for pair in split_data:
        pairs.add((pair[0], pair[1]))
        pairs.add((pair[1], pair[0]))

    rows = []

    # ---- Add real (original) videos ----
    # Collect unique video IDs from all pairs
    for video_id in sorted(
        {x for pair in split_data for x in pair}
    ):
        path = DATASETS["original"]["path"] / f"{video_id}.mp4"

        # Only include if the video file actually exists on disk
        if path.exists():
            rows.append({
                "video_path": str(path),
                "dataset":    "original",
                "video_id":   video_id,
                "label":      0   # Real
            })

    # ---- Add manipulated (fake) videos ----
    # For each fake dataset and each source-target pair, check if the video exists
    for dataset in [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures"
    ]:
        folder = DATASETS[dataset]["path"]

        for src, dst in sorted(pairs):
            # Manipulated videos are named like: 001_002.mp4
            path = folder / f"{src}_{dst}.mp4"

            if path.exists():
                rows.append({
                    "video_path": str(path),
                    "dataset":    dataset,
                    "video_id":   f"{src}_{dst}",
                    "label":      1   # Fake
                })

    # ---- Write manifest CSV ----
    output_file = OUTPUT / f"{split_name}.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "dataset", "video_id", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{split_name}: {len(rows)} videos")
    print(f"Saved: {output_file}")


# ---- Run for all three splits ----
for split in ["train", "val", "test"]:
    process_split(split)

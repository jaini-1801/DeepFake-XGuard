import json
import csv
from pathlib import Path

ROOT = Path(r"E:\DeepFakeX")
DATA = ROOT / "data"
SPLITS = DATA / "splits"
OUTPUT = DATA / "manifests"

OUTPUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "original": {
        "path": DATA / "original_sequences" / "youtube" / "c40" / "videos",
        "label": 0
    },
    "Deepfakes": {
        "path": DATA / "manipulated_sequences" / "Deepfakes" / "c40" / "videos",
        "label": 1
    },
    "Face2Face": {
        "path": DATA / "manipulated_sequences" / "Face2Face" / "c40" / "videos",
        "label": 1
    },
    "FaceShifter": {
        "path": DATA / "manipulated_sequences" / "FaceShifter" / "c40" / "videos",
        "label": 1
    },
    "FaceSwap": {
        "path": DATA / "manipulated_sequences" / "FaceSwap" / "c40" / "videos",
        "label": 1
    },
    "NeuralTextures": {
        "path": DATA / "manipulated_sequences" / "NeuralTextures" / "c40" / "videos",
        "label": 1
    }
}


def load_split(name):
    with open(SPLITS / f"{name}.json", "r") as f:
        return json.load(f)


def process_split(split_name):
    split_data = load_split(split_name)

    # Create all source-target pairs from the official split
    pairs = set()

    for pair in split_data:
        pairs.add((pair[0], pair[1]))
        pairs.add((pair[1], pair[0]))

    rows = []

    # Original videos
    for video_id in sorted(
        {x for pair in split_data for x in pair}
    ):
        path = DATASETS["original"]["path"] / f"{video_id}.mp4"

        if path.exists():
            rows.append({
                "video_path": str(path),
                "dataset": "original",
                "video_id": video_id,
                "label": 0
            })

    # Manipulated videos
    for dataset in [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures"
    ]:
        folder = DATASETS[dataset]["path"]

        for src, dst in sorted(pairs):
            path = folder / f"{src}_{dst}.mp4"

            if path.exists():
                rows.append({
                    "video_path": str(path),
                    "dataset": dataset,
                    "video_id": f"{src}_{dst}",
                    "label": 1
                })

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


for split in ["train", "val", "test"]:
    process_split(split)
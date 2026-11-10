import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"E:\DeepFakeX")
FRAMES_DIR = ROOT / "data" / "frames"

FRAMES_PER_VIDEO = 10

for split in ["train", "val", "test"]:

    csv_path = ROOT / "data" / "manifests" / f"{split}.csv"
    df = pd.read_csv(csv_path)

    split_dir = FRAMES_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {split}: {len(df)} videos")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):

        video_path = Path(row["video_path"])
        dataset = row["dataset"]
        video_id = row["video_id"]

        output_dir = split_dir / dataset / str(video_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already processed
        existing = list(output_dir.glob("*.jpg"))
        if len(existing) >= FRAMES_PER_VIDEO:
            continue

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Could not open: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            continue

        # Select evenly spaced frames
        frame_indices = [
            int(x)
            for x in __import__("numpy").linspace(
                0, total_frames - 1, FRAMES_PER_VIDEO
            )
        ]

        for i, frame_idx in enumerate(frame_indices):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            success, frame = cap.read()

            if not success:
                continue

            output_path = output_dir / f"frame_{i:02d}.jpg"

            cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

        cap.release()

print("\nFrame extraction completed.")
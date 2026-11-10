import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"E:\DeepFakeX")

MANIFEST = ROOT / "Data" / "manifests" / "celebdf_official_test.csv"
FRAMES_DIR = ROOT / "Data" / "frames" / "celebdf_test"

FRAMES_PER_VIDEO = 10

df = pd.read_csv(MANIFEST)

print(f"Total Celeb-DF test videos: {len(df)}")

processed = 0
failed = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting frames"):

    video_path = Path(row["video_path"])
    category = row["category"]

    video_id = video_path.stem
    output_dir = FRAMES_DIR / category / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*.jpg"))

    if len(existing) >= FRAMES_PER_VIDEO:
        processed += 1
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"\nCould not open: {video_path}")
        failed += 1
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        failed += 1
        continue

    frame_indices = np.linspace(
        0, total_frames - 1,
        FRAMES_PER_VIDEO,
        dtype=int
    )

    saved = 0

    for i, frame_idx in enumerate(frame_indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        success, frame = cap.read()

        if not success:
            continue

        output_path = output_dir / f"frame_{i:02d}.jpg"

        cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        saved += 1

    cap.release()

    if saved == 0:
        failed += 1
    else:
        processed += 1

print("\nFrame extraction completed.")
print("Videos processed:", processed)
print("Videos failed:", failed)
print("Output:", FRAMES_DIR)

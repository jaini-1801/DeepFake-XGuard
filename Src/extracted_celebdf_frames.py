"""
extracted_celebdf_frames.py
---------------------------
Extracts frames from Celeb-DF test videos.

This is the Celeb-DF equivalent of extract_frames.py.
It reads the official Celeb-DF test manifest (CSV file) and
extracts FRAMES_PER_VIDEO evenly-spaced frames from each video.

Folder structure created:
    Data/frames/celebdf_test/
        Celeb-real/        <- real celebrity videos
            {video_id}/    <- one folder per video
                frame_00.jpg
                frame_01.jpg
                ...
        Celeb-synthesis/   <- deepfake celebrity videos
        YouTube-real/      <- real YouTube videos

Input CSV columns:
  - video_path: full path to the .mp4 file
  - category:   e.g. 'Celeb-real', 'Celeb-synthesis'
  - label:      0 = Real, 1 = Fake
"""

import cv2             # For reading video files and extracting frames
import pandas as pd    # For reading the CSV manifest
import numpy as np     # For evenly spacing frame indices
from pathlib import Path
from tqdm import tqdm  # Progress bar in the terminal

ROOT = Path(r"E:\DeepFakeX")

# CSV file provided by the Celeb-DF authors listing all test videos
MANIFEST   = ROOT / "Data" / "manifests" / "celebdf_official_test.csv"

# Where to save the extracted frames
FRAMES_DIR = ROOT / "Data" / "frames" / "celebdf_test"

FRAMES_PER_VIDEO = 10  # How many frames to extract per video

# Load the manifest
df = pd.read_csv(MANIFEST)
print(f"Total Celeb-DF test videos: {len(df)}")

processed = 0  # Videos successfully processed
failed    = 0  # Videos that could not be opened or had no frames

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting frames"):

    video_path = Path(row["video_path"])
    category   = row["category"]           # e.g. 'Celeb-real' or 'Celeb-synthesis'

    # Use the filename (without extension) as the unique video ID
    video_id   = video_path.stem           # e.g. 'id0_id1_0001'
    output_dir = FRAMES_DIR / category / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip this video if it was already processed
    existing = list(output_dir.glob("*.jpg"))
    if len(existing) >= FRAMES_PER_VIDEO:
        processed += 1
        continue

    # Open the video file
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

    # Choose FRAMES_PER_VIDEO evenly spaced frame positions
    # e.g. for a 100-frame video: [0, 11, 22, 33, ... 99]
    frame_indices = np.linspace(
        0, total_frames - 1,
        FRAMES_PER_VIDEO,
        dtype=int
    )

    saved = 0

    for i, frame_idx in enumerate(frame_indices):

        # Jump to this frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        success, frame = cap.read()

        if not success:
            continue  # Couldn't read this frame, try the next one

        output_path = output_dir / f"frame_{i:02d}.jpg"

        # Save as JPEG with quality 95 (high quality)
        cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        saved += 1

    cap.release()  # Always release the video file handle

    if saved == 0:
        failed += 1
    else:
        processed += 1

print("\nFrame extraction completed.")
print("Videos processed:", processed)
print("Videos failed:", failed)
print("Output:", FRAMES_DIR)

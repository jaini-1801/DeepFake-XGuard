"""
extract_frames.py
-----------------
Step 1 of the data pipeline: extract frames (still images) from videos.

For each video listed in the CSV manifests (train/val/test),
this script samples FRAMES_PER_VIDEO evenly-spaced frames and saves
them as .jpg files. These frames are later used to extract face crops.

Folder structure created:
    data/frames/
        train/
            original/   <- real video frames
            Deepfakes/  <- deepfake frames
            ...
        val/
        test/
"""

import cv2             # OpenCV: used to read video files and extract frames
import pandas as pd    # For reading the CSV manifest files
import numpy as np     # For evenly spacing frame indices
import gc              # Garbage collector: frees memory after each video
from pathlib import Path
from tqdm import tqdm  # Shows a progress bar in the terminal

# ---- Config ----
ROOT = Path(r"E:\DeepFakeX")
FRAMES_DIR = ROOT / "data" / "frames"   # Where extracted frames will be saved

FRAMES_PER_VIDEO = 10   # How many frames to extract per video

# Limit OpenCV to 1 thread to reduce memory overhead (safe for sequential processing)
cv2.setNumThreads(1)

# ---- Process each data split ----
for split in ["train", "val", "test"]:

    # Read the manifest CSV which lists all videos for this split
    csv_path = ROOT / "data" / "manifests" / f"{split}.csv"
    df = pd.read_csv(csv_path)

    # Create output directory for this split
    split_dir = FRAMES_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {split}: {len(df)} videos")

    failed_videos = []  # Track videos that couldn't be processed

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):

        video_path = Path(row["video_path"])
        dataset    = row["dataset"]    # e.g. 'original', 'Deepfakes', 'Face2Face'
        video_id   = str(row["video_id"])

        # Save frames under: frames/{split}/{dataset}/{video_id}/
        output_dir = split_dir / dataset / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Skip this video if we already extracted enough frames from it
        existing = list(output_dir.glob("*.jpg"))
        if len(existing) >= FRAMES_PER_VIDEO:
            continue

        cap = None  # Video capture object (opened below)

        try:
            # Open the video file
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                # Could not open the video (file missing or corrupted)
                failed_videos.append(str(video_path))
                continue

            # Total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                # Couldn't read frame count -- skip this video
                failed_videos.append(str(video_path))
                continue

            # Pick FRAMES_PER_VIDEO evenly spaced indices across the video
            # e.g. for a 100-frame video and 10 frames: [0, 11, 22, 33, ... 99]
            frame_indices = np.linspace(
                0, total_frames - 1, FRAMES_PER_VIDEO
            ).astype(int)

            success_count = 0

            for i, frame_idx in enumerate(frame_indices):

                # Jump to the desired frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

                # Read the frame
                success, frame = cap.read()

                if not success:
                    continue  # Couldn't read this frame, skip it

                output_path = output_dir / f"frame_{i:02d}.jpg"

                # Save frame as JPEG with quality=95 (high quality)
                saved = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )

                if saved:
                    success_count += 1

                del frame  # Free memory immediately after saving

            # If we couldn't extract all frames, mark this video as failed
            if success_count < FRAMES_PER_VIDEO:
                failed_videos.append(str(video_path))

        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            failed_videos.append(str(video_path))

        finally:
            # Always release the video capture object, even if an error occurred
            if cap is not None:
                cap.release()

            gc.collect()  # Free any leftover memory

    print(f"\n{split} extraction completed.")
    print("Failed videos:", len(failed_videos))

    # Save the list of failed videos to a file for inspection
    if failed_videos:
        failed_file = ROOT / "results" / f"{split}_failed_videos.txt"
        failed_file.parent.mkdir(parents=True, exist_ok=True)
        failed_file.write_text("\n".join(failed_videos))
        print("Failed video list saved:", failed_file)

print("\nAll split extraction attempts completed.")

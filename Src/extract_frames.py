
import cv2
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"E:\DeepFakeX")
FRAMES_DIR = ROOT / "data" / "frames"

FRAMES_PER_VIDEO = 10

# Reduce OpenCV memory/thread overhead
cv2.setNumThreads(1)

for split in ["train", "val", "test"]:

    csv_path = ROOT / "data" / "manifests" / f"{split}.csv"
    df = pd.read_csv(csv_path)

    split_dir = FRAMES_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {split}: {len(df)} videos")

    failed_videos = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):

        video_path = Path(row["video_path"])
        dataset = row["dataset"]
        video_id = str(row["video_id"])

        output_dir = split_dir / dataset / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Skip fully processed videos
        existing = list(output_dir.glob("*.jpg"))
        if len(existing) >= FRAMES_PER_VIDEO:
            continue

        cap = None

        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                failed_videos.append(str(video_path))
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                failed_videos.append(str(video_path))
                continue

            frame_indices = np.linspace(
                0, total_frames - 1, FRAMES_PER_VIDEO
            ).astype(int)

            success_count = 0

            for i, frame_idx in enumerate(frame_indices):

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

                success, frame = cap.read()

                if not success:
                    continue

                output_path = output_dir / f"frame_{i:02d}.jpg"

                saved = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )

                if saved:
                    success_count += 1

                del frame

            if success_count < FRAMES_PER_VIDEO:
                failed_videos.append(str(video_path))

        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            failed_videos.append(str(video_path))

        finally:
            if cap is not None:
                cap.release()

            gc.collect()

    print(f"\n{split} extraction completed.")
    print("Failed videos:", len(failed_videos))

    if failed_videos:
        failed_file = ROOT / "results" / f"{split}_failed_videos.txt"
        failed_file.parent.mkdir(parents=True, exist_ok=True)
        failed_file.write_text("\n".join(failed_videos))

        print("Failed video list saved:", failed_file)

print("\nAll split extraction attempts completed.")

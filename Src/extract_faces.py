"""
extract_faces.py
----------------
Step 2 of the data pipeline: detect and crop faces from extracted frames.

Uses MTCNN (Multi-task Cascaded Convolutional Network) to find faces
in each frame image, then saves the cropped face as a 224x224 JPEG.

Why faces only?
  The deepfake manipulation is applied to the face region, so cropping
  faces removes irrelevant background and helps the model focus on
  the right features.

Input:  data/frames/{split}/{dataset}/{video_id}/frame_XX.jpg
Output: data/faces/{split}/{dataset}/{video_id}/frame_XX.jpg
"""

import torch
import gc                              # Garbage collector for freeing memory
from facenet_pytorch import MTCNN     # Pre-trained face detection model
from PIL import Image
from pathlib import Path
from tqdm import tqdm                 # Progress bar

# ---- Config ----
ROOT        = Path(r"E:\DeepFakeX")
FRAMES_DIR  = ROOT / "data" / "frames"   # Input: raw video frames
FACES_DIR   = ROOT / "data" / "faces"    # Output: cropped face images
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use GPU for faster face detection if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---- MTCNN face detector setup ----
# image_size=224: output face crop will be 224x224 pixels
# margin=20: add a 20-pixel border around the detected face box
# min_face_size=40: ignore faces smaller than 40x40 pixels (too blurry)
# keep_all=False: keep only the largest/most confident face per image
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=40,
    keep_all=False,
    device=device
)

# ---- Process each data split ----
for split in ["train", "val", "test"]:

    input_dir  = FRAMES_DIR / split
    output_dir = FACES_DIR  / split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all frame images in this split
    frame_files = list(input_dir.rglob("*.jpg"))

    print(f"\n{split}: {len(frame_files)} frames")

    failed  = []    # Frames where no face was found or an error occurred
    detected = 0    # Count of successfully extracted faces
    skipped  = 0    # Count of frames already processed (skip to save time)

    for frame_path in tqdm(frame_files, desc=f"Faces - {split}"):

        # Reproduce the same relative path in the output directory
        # e.g. frames/train/Deepfakes/001/frame_00.jpg
        #   -> faces/train/Deepfakes/001/frame_00.jpg
        relative    = frame_path.relative_to(input_dir)
        output_path = output_dir / relative

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if the face image already exists and is non-empty (resume support)
        if output_path.exists() and output_path.stat().st_size > 0:
            skipped += 1
            continue

        image = None
        face  = None

        try:
            # Open frame image
            image = Image.open(frame_path).convert("RGB")

            # Run face detection (no gradient needed -- inference only)
            with torch.no_grad():
                face = mtcnn(image)

            # If MTCNN didn't find a face, skip this frame
            if face is None:
                failed.append(str(frame_path))
                continue

            # MTCNN returns a tensor with values in [-1, 1]
            # Convert to [0, 255] uint8 for saving as JPEG
            face = ((face + 1) / 2 * 255).clamp(0, 255)
            face = face.byte()

            # Rearrange from (C, H, W) tensor to (H, W, C) numpy array for PIL
            face_image = face.permute(1, 2, 0).cpu().numpy()

            # Save the cropped face as a JPEG
            Image.fromarray(face_image).save(
                output_path,
                format="JPEG",
                quality=95
            )

            detected += 1

        except Exception as e:
            print(f"\nError: {frame_path} -> {e}")
            failed.append(str(frame_path))

            # On GPU errors, clear the cache to avoid running out of memory
            if device == "cuda":
                torch.cuda.empty_cache()

        finally:
            # Clean up regardless of success or failure
            if image is not None:
                image.close()

            del image, face
            gc.collect()

    # Save list of failed frames to file for review
    failed_file = RESULTS_DIR / f"{split}_face_failures.txt"
    failed_file.write_text("\n".join(failed))

    print(f"\n{split} completed")
    print("Faces detected:", detected)
    print("Already existing:", skipped)
    print("Failed/no detection:", len(failed))
    print("Failure log:", failed_file)

    # Clear GPU memory between splits
    if device == "cuda":
        torch.cuda.empty_cache()

print("\nAll face extraction attempts completed.")

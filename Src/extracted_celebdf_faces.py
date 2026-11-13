"""
extracted_celebdf_faces.py
--------------------------
Extracts face crops from Celeb-DF test frames using MTCNN.

This is the Celeb-DF equivalent of extract_faces.py.
Reads frames extracted by extracted_celebdf_frames.py and
detects + crops the face region from each one using MTCNN.

Why crop faces?
  Deepfake manipulation is applied to faces only. Cropping removes
  irrelevant background and helps the detection model focus on
  the manipulated region.

Input:  Data/frames/celebdf_test/{category}/{video_id}/frame_XX.jpg
Output: Data/faces/celebdf_test/{category}/{video_id}/frame_XX.jpg

Skips frames that already have a saved face (safe to re-run after a crash).
"""

import torch
from facenet_pytorch import MTCNN   # Pre-trained multi-stage face detector
from PIL import Image
from pathlib import Path
from tqdm import tqdm               # Progress bar

ROOT = Path(r"E:\DeepFakeX")

FRAMES_DIR = ROOT / "Data" / "frames" / "celebdf_test"  # Input: raw frames
FACES_DIR  = ROOT / "Data" / "faces"  / "celebdf_test"  # Output: face crops

# Use GPU for faster face detection if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---- MTCNN face detector ----
# image_size=224: output face crop will be 224x224 pixels
# margin=20:      add 20 pixels of padding around the detected face box
# min_face_size=40: ignore very small faces (too low resolution to be useful)
# keep_all=False: keep only the most prominent face per image
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=40,
    keep_all=False,
    device=device
)

# Collect all frame files from the Celeb-DF test folder (recursively)
frame_files = list(FRAMES_DIR.rglob("*.jpg"))
print("Total frames:", len(frame_files))

saved   = 0   # Frames where a face was successfully detected and saved
no_face = 0   # Frames where MTCNN found no face
errors  = 0   # Frames that caused an unexpected error

for frame_path in tqdm(frame_files, desc="Extracting faces"):

    # Reproduce the same subfolder structure in the output directory
    # e.g. frames/celebdf_test/Celeb-real/vid001/frame_00.jpg
    #   -> faces/celebdf_test/Celeb-real/vid001/frame_00.jpg
    relative    = frame_path.relative_to(FRAMES_DIR)
    output_path = FACES_DIR / relative

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if face already extracted (allows resuming after a crash)
    if output_path.exists():
        continue

    try:
        # Open frame image
        image = Image.open(frame_path).convert("RGB")

        # Run MTCNN face detection
        # Returns a tensor of shape (3, 224, 224) with values in [-1, 1]
        # or None if no face found
        face = mtcnn(image)

        if face is None:
            # No face detected in this frame -- skip it
            no_face += 1
            continue

        # Convert from MTCNN's [-1, 1] range to [0, 255] uint8 for saving as JPEG
        face = ((face + 1) / 2 * 255).clamp(0, 255)
        face = face.byte()

        # Rearrange tensor from (C, H, W) to (H, W, C) numpy format for PIL
        face_image = face.permute(1, 2, 0).cpu().numpy()

        # Save the cropped face as JPEG
        Image.fromarray(face_image).save(output_path, quality=95)

        saved += 1

    except Exception as e:
        errors += 1
        print(f"\nError: {frame_path} -> {e}")

print("\nFace extraction completed.")
print("Faces saved:", saved)
print("Frames without faces:", no_face)
print("Errors:", errors)
print("Output:", FACES_DIR)

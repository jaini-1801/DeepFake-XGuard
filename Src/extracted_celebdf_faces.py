
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"E:\DeepFakeX")

FRAMES_DIR = ROOT / "Data" / "frames" / "celebdf_test"
FACES_DIR = ROOT / "Data" / "faces" / "celebdf_test"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=40,
    keep_all=False,
    device=device
)

frame_files = list(FRAMES_DIR.rglob("*.jpg"))

print("Total frames:", len(frame_files))

saved = 0
no_face = 0
errors = 0

for frame_path in tqdm(frame_files, desc="Extracting faces"):

    relative = frame_path.relative_to(FRAMES_DIR)
    output_path = FACES_DIR / relative

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        continue

    try:
        image = Image.open(frame_path).convert("RGB")

        face = mtcnn(image)

        if face is None:
            no_face += 1
            continue

        face = ((face + 1) / 2 * 255).clamp(0, 255)
        face = face.byte()

        face_image = face.permute(1, 2, 0).cpu().numpy()

        Image.fromarray(face_image).save(
            output_path,
            quality=95
        )

        saved += 1

    except Exception as e:
        errors += 1
        print(f"\nError: {frame_path} -> {e}")

print("\nFace extraction completed.")
print("Faces saved:", saved)
print("Frames without faces:", no_face)
print("Errors:", errors)
print("Output:", FACES_DIR)

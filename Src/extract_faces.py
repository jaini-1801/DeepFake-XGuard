import torch
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"E:\DeepFakeX")
FRAMES_DIR = ROOT / "data" / "frames"
FACES_DIR = ROOT / "data" / "faces"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=40,
    keep_all=False,
    device=device
)

for split in ["train", "val", "test"]:

    input_dir = FRAMES_DIR / split
    output_dir = FACES_DIR / split

    frame_files = list(input_dir.rglob("*.jpg"))

    print(f"\n{split}: {len(frame_files)} frames")

    for frame_path in tqdm(frame_files, desc=f"Faces - {split}"):

        relative = frame_path.relative_to(input_dir)
        output_path = output_dir / relative

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            continue

        try:
            image = Image.open(frame_path).convert("RGB")

            face = mtcnn(image)

            if face is None:
                continue

            # Convert tensor [-1,1] → [0,255]
            face = ((face + 1) / 2 * 255).clamp(0, 255)
            face = face.byte()

            face_image = face.permute(1, 2, 0).cpu().numpy()

            Image.fromarray(face_image).save(
                output_path,
                quality=95
            )

        except Exception as e:
            print(f"Error: {frame_path} -> {e}")

print("\nFace extraction completed.")
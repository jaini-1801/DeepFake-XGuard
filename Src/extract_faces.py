
import torch
import gc
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"E:\DeepFakeX")
FRAMES_DIR = ROOT / "data" / "frames"
FACES_DIR = ROOT / "data" / "faces"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = list(input_dir.rglob("*.jpg"))

    print(f"\n{split}: {len(frame_files)} frames")

    failed = []
    detected = 0
    skipped = 0

    for frame_path in tqdm(frame_files, desc=f"Faces - {split}"):

        relative = frame_path.relative_to(input_dir)
        output_path = output_dir / relative

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Reuse successfully extracted faces
        if output_path.exists() and output_path.stat().st_size > 0:
            skipped += 1
            continue

        image = None
        face = None

        try:
            image = Image.open(frame_path).convert("RGB")

            with torch.no_grad():
                face = mtcnn(image)

            if face is None:
                failed.append(str(frame_path))
                continue

            # Convert MTCNN output from [-1, 1] to [0, 255]
            face = ((face + 1) / 2 * 255).clamp(0, 255)
            face = face.byte()

            face_image = face.permute(1, 2, 0).cpu().numpy()

            Image.fromarray(face_image).save(
                output_path,
                format="JPEG",
                quality=95
            )

            detected += 1

        except Exception as e:
            print(f"\nError: {frame_path} -> {e}")
            failed.append(str(frame_path))

            if device == "cuda":
                torch.cuda.empty_cache()

        finally:
            if image is not None:
                image.close()

            del image, face
            gc.collect()

    failed_file = RESULTS_DIR / f"{split}_face_failures.txt"
    failed_file.write_text("\n".join(failed))

    print(f"\n{split} completed")
    print("Faces detected:", detected)
    print("Already existing:", skipped)
    print("Failed/no detection:", len(failed))
    print("Failure log:", failed_file)

    if device == "cuda":
        torch.cuda.empty_cache()

print("\nAll face extraction attempts completed.")

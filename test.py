# Requires Pillow: pip install pillow

from pathlib import Path
from PIL import Image

COVER_DIR = Path("data/covers")
print(f"Looking for image files in: {COVER_DIR.resolve()}")
# Gather all JPG and JPEG files, case-insensitive
patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
files = []
for pattern in patterns:
    files.extend(COVER_DIR.glob(pattern))
if not files:
    print("No image files found in the directory.")
for img_path in files:
    print(f"Processing file: {img_path.name}")
    with Image.open(img_path) as img:
        print(f"{img_path.name}: {img.width}×{img.height} px, mode={img.mode}")

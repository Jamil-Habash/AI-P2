import os
from pathlib import Path
from PIL import Image

# Input and output folders
input_root = Path("brain")
output_root = Path("Brain Dataset Resized")
target_size = (32, 32)

# Traverse training and testing folders
for split in ["test", "train"]:
    input_dir = input_root / split
    output_dir = output_root / split

    for class_folder in input_dir.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for img_file in class_folder.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                with Image.open(img_file) as img:
                    img = img.convert("RGB")
                    img = img.resize(target_size)
                    output_path = output_class_dir / img_file.name
                    img.save(output_path)
                    print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed: {img_file} — {e}")
import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Path to the resized dataset
dataset_path = Path("Brain Dataset Resized")
image_size = (32, 32)
flattened_size = 32 * 32 * 3

X = []
y = []

# Loop through training and testing sets
for split in ["train", "test"]:
    split_path = dataset_path / split
    for class_folder in split_path.iterdir():
        if not class_folder.is_dir():
            continue
        label = class_folder.name
        for img_file in class_folder.glob("*.*"):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            try:
                with Image.open(img_file) as img:
                    img = img.convert("RGB")
                    img = img.resize(image_size)  # Safety resize
                    img_array = np.array(img).flatten()
                    X.append(img_array)
                    y.append(label)
            except Exception as e:
                print(f"⚠Error processing {img_file}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the arrays
np.save("X_flattened.npy", X)
np.save("y_labels_encoded.npy", y_encoded)
np.save("label_classes.npy", label_encoder.classes_)

print("Done!")
print("X shape:", X.shape)
print("y shape:", y_encoded.shape)
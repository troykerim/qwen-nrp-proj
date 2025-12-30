import os
import shutil

# Input and output directories
INPUT_DIR = "/home/troy/Downloads/Jam Causing Material-dataset.voc/valid"
IMG_OUT_DIR = "/home/troy/jam-causing-material/valid/images"
LBL_OUT_DIR = "/home/troy/jam-causing-material/valid/labels"

# Create output directories if they don't exist
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(LBL_OUT_DIR, exist_ok=True)

jpg_count = 0
xml_count = 0

for filename in os.listdir(INPUT_DIR):
    src_path = os.path.join(INPUT_DIR, filename)

    # Skip subdirectories if any exist
    if not os.path.isfile(src_path):
        continue

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".jpg":
        shutil.copy2(src_path, os.path.join(IMG_OUT_DIR, filename))
        jpg_count += 1

    elif ext == ".xml":
        shutil.copy2(src_path, os.path.join(LBL_OUT_DIR, filename))
        xml_count += 1

print(f"Done.")
print(f"Copied {jpg_count} JPG files to: {IMG_OUT_DIR}")
print(f"Copied {xml_count} XML files to: {LBL_OUT_DIR}")
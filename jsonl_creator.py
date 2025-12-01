'''
    JSONL file formater for formatting the Dataset into string tokens that are accepted by Qwen 
'''
import os
import json
from pathlib import Path

BASE = "/home/troy/qwen-nrp-proj/dataset"

TRAIN_IMG_DIR = f"{BASE}/train/images"
TRAIN_LBL_DIR = f"{BASE}/train/labels"

VALID_IMG_DIR = f"{BASE}/valid/images"
VALID_LBL_DIR = f"{BASE}/valid/labels"

# Only run Train or Valid, not at the same time
# OUT_TRAIN = "/home/troy/qwen-nrp-proj/train.jsonl"  
OUT_VALID = "/home/troy/qwen-nrp-proj/valid.jsonl"  
NAUTILUS_BASE = "/workspace/data/dataset"

# Parse Pascal VOC label format
# Format: ClassName xmin xmax ymin ymax
def load_voc_labels(label_path):
    entries = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xmin, xmax, ymin, ymax = parts
            entries.append(f"{cls} {xmin} {xmax} {ymin} {ymax}")

    return entries

def to_nautilus_path(local_img_path):

    local_suffix = local_img_path.split("/dataset/", 1)[1]
    return f"{NAUTILUS_BASE}/{local_suffix}"

def build_jsonl_entry(image_path_local, bbox_lines):

    # Convert path inside JSONL to Nautilus path
    image_path = to_nautilus_path(image_path_local)

    system_msg = {
        "role": "system",
        "content":
            "You are an assistant that detects waste objects in images. "
            "The possible waste categories are: Glass-A, Green waste-A, Metal, "
            "Organics-A, Organics-B-NOT, Organics-E, Others, Paper-A, Paper-B, Paper-D, "
            "Plastic-A, Plastic-B, Plastic-C, Plastic-D, Plastic-E, Plastic-G, Wood."
    }

    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",
             "text": "Detect the waste objects in this image and output their bounding boxes in the format: class_name xmin xmax ymin ymax"},
        ],
    }

    assistant_msg = {
        "role": "assistant",
        "content": "\n".join(bbox_lines)
    }

    return {"messages": [system_msg, user_msg, assistant_msg]}


# Format the JSONL file 
def generate_jsonl(img_dir, lbl_dir, output_path):

    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    entries = []

    for img_name in image_files:
        stem = Path(img_name).stem
        lbl_path = f"{lbl_dir}/{stem}.txt"

        if not os.path.exists(lbl_path):
            continue

        bbox_lines = load_voc_labels(lbl_path)
        if not bbox_lines:
            continue

        # LOCAL PC path
        local_img_path = f"{img_dir}/{img_name}"

        entry = build_jsonl_entry(local_img_path, bbox_lines)
        entries.append(entry)

    with open(output_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"Created {len(entries)} entries → {output_path}")
    
# For train.jsonl 
# print("Generating train.jsonl...")
#generate_jsonl(TRAIN_IMG_DIR, TRAIN_LBL_DIR, OUT_TRAIN)

# Uncomment for valid.jsonl
print("Generating valid.jsonl...")
generate_jsonl(VALID_IMG_DIR, VALID_LBL_DIR, OUT_VALID)


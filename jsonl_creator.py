import os
import json
from pathlib import Path

# -------------------------------------------------------------------
# GOOGLE COLAB + GOOGLE DRIVE PATHS
# -------------------------------------------------------------------
DATASET_DIR = "/home/troy/qwen-nrp-proj/dataset"

TRAIN_IMG_DIR = f"{DATASET_DIR}/train/images"
TRAIN_LBL_DIR = f"{DATASET_DIR}/train/labels"

# Output JSONL files (focus on train first)
TRAIN_JSONL = "/home/troy/qwen-nrp-proj/train.jsonl"
VALID_JSONL = "/home/troy/qwen-nrp-proj/valid.jsonl" # Not used yet


# -------------------------------------------------------------------
# Parse Pascal VOC text label file
# Format: class_name xmin ymin xmax ymax
# -------------------------------------------------------------------
def load_voc_label(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xmin, ymin, xmax, ymax = parts

            boxes.append({
                "class_name": cls,
                "xmin": int(xmin),
                "ymin": int(ymin),
                "xmax": int(xmax),
                "ymax": int(ymax)
            })

    return boxes


# -------------------------------------------------------------------
# Build JSONL entry according to Qwen2.5-VL supervised tuning format
# -------------------------------------------------------------------
def build_jsonl_entry(image_path, boxes):
    messages = []

    # System prompt
    messages.append({
        "role": "system",
        "content": "You are an AI assistant that identifies objects and outputs bounding boxes in <bbox> format."
    })

    # User message containing the image
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Identify all objects in the image and output bounding boxes."}
        ]
    })

    # Assistant message containing bounding boxes in tag format
    bbox_list = []
    for b in boxes:
        bbox_list.append(
            f"<bbox class='{b['class_name']}' "
            f"xmin='{b['xmin']}' ymin='{b['ymin']}' "
            f"xmax='{b['xmax']}' ymax='{b['ymax']}'></bbox>"
        )

    messages.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": " ".join(bbox_list)}
        ]
    })

    return {"messages": messages}


# -------------------------------------------------------------------
# Process a split (train or valid)
# -------------------------------------------------------------------
def process_split(img_dir, lbl_dir, output_path):
    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    jsonl_entries = []

    for img_name in img_files:
        stem = Path(img_name).stem
        lbl_path = f"{lbl_dir}/{stem}.txt"

        if not os.path.exists(lbl_path):
            continue

        boxes = load_voc_label(lbl_path)
        if not boxes:
            continue

        # IMPORTANT: Use Google Drive path for Qwen VL training
        full_img_path = f"{img_dir}/{img_name}"

        entry = build_jsonl_entry(full_img_path, boxes)
        jsonl_entries.append(entry)

    # Write JSONL file
    with open(output_path, "w") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Created {len(jsonl_entries)} entries → {output_path}")


# -------------------------------------------------------------------
# MAIN: Only train.jsonl first
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating train.jsonl ...")
    process_split(TRAIN_IMG_DIR, TRAIN_LBL_DIR, TRAIN_JSONL)

    # ---- Uncomment this block when ready to generate valid.jsonl ----
    # print("Generating valid.jsonl ...")
    # process_split(
    #     f"{DATASET_DIR}/valid/images",
    #     f"{DATASET_DIR}/valid/labels",
    #     VALID_JSONL
    # )

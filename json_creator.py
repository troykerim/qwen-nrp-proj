import os
import json


# train_labels_dir = "/home/troy/qwen-nrp-proj/dataset/train/labels"
# train_output_jsonl = "/home/troy/qwen-nrp-proj/train.jsonl"
# train_image_dir_in_pvc = "/workspace/data/dataset/train/images"

# --- Uncomment for validation set ---
valid_labels_dir = "/home/troy/qwen-nrp-proj/dataset/valid/labels"
valid_output_jsonl = "/home/troy/qwen-nrp-proj/valid.jsonl"
valid_image_dir_in_pvc = "/workspace/data/dataset/valid/images"


def label_file_to_entry(label_path, image_dir_pvc):
    """
    Convert one label .txt file (already preprocessed from Pascal VOC)
    into a LLaVA-style JSON entry for Qwen2.5-VL fine-tuning.
    """

    # Match label file name to corresponding image
    image_name = os.path.splitext(os.path.basename(label_path))[0] + ".jpg"
    image_uri = f"file://{os.path.join(image_dir_pvc, image_name)}"

    # Read all label lines (keep them exactly as-is)
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return None  # Skip empty files

    # --- Format in LLaVA/Qwen style ---
    user_prompt = "<image>\nDescribe the waste items and their bounding boxes."
    assistant_response = "\n".join(lines)

    return {
        "id": os.path.basename(label_path),
        "image": image_uri,
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": assistant_response}
        ]
    }


def convert_labels_to_jsonl(labels_dir, output_jsonl, image_dir_pvc):
    """Convert all .txt label files in one folder to a single JSONL file."""
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
    count = 0

    with open(output_jsonl, "w") as out_f:
        for file_name in label_files:
            label_path = os.path.join(labels_dir, file_name)
            entry = label_file_to_entry(label_path, image_dir_pvc)
            if entry:
                json.dump(entry, out_f, ensure_ascii=False)
                out_f.write("\n")
                count += 1

    print(f"Created {output_jsonl} with {count} entries.")

if __name__ == "__main__":
    # --- Train set (uncomment to run for training set only) ---
    # convert_labels_to_jsonl(train_labels_dir, train_output_jsonl, train_image_dir_in_pvc)

    # --- Validation set (uncomment to run for validation set only) ---
    convert_labels_to_jsonl(valid_labels_dir, valid_output_jsonl, valid_image_dir_in_pvc)

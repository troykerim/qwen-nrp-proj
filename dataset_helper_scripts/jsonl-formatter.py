'''
Python script to formatter to create the JSONL files to be used by Qwen and Nautilus.
Edit the file path so that it reads from the correct folder {train, valid or test} and creates train.jsonl, valid.jsonl...

This script will read each .txt file from respective label folder and then input the bounding box information in the correct location
It will also read the file name and insert that file name in the correct location.
'''
import os
import json
from pathlib import Path

LABEL_DIR = Path("/home/troy/jam-causing-material/valid/labels")
OUTPUT_JSONL = Path("/home/troy/qwen-nrp-proj/valid.jsonl")

IMAGE_BASE_PATH = "/workspace/data/jam-causing-material/valid/images"   # Dataset Path for Nautilus

SYSTEM_PROMPT = (
    "You are an assistant that detects jam causing objects in images. "
    "The possible jam causing objects are: fabrics, rigid-plastic, "
    "non-recyclables, large-plastic-films, unopened-plastic-bags, "
    "metal, wrappables, wood."
)

USER_TEXT_PROMPT = (
    "Detect the waste objects in this image and output their bounding boxes "
    "in the format: class_name xmin ymin ymax xmax"
)

def build_jsonl():
    label_files = sorted(LABEL_DIR.glob("*.txt"))

    with open(OUTPUT_JSONL, "w") as out_f:
        for label_path in label_files:
            # Image filename derived from label filename
            image_name = label_path.stem + ".jpg"
            image_path = f"{IMAGE_BASE_PATH}/{image_name}"

            # Read bounding box lines
            with open(label_path, "r") as f:
                bbox_lines = [line.strip() for line in f if line.strip()]

            assistant_content = "\n".join(bbox_lines)

            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path
                            },
                            {
                                "type": "text",
                                "text": USER_TEXT_PROMPT
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    }
                ]
            }

            out_f.write(json.dumps(entry) + "\n")

    print(f"Created JSONL with {len(label_files)} entries:")
    print(OUTPUT_JSONL)


if __name__ == "__main__":
    build_jsonl()

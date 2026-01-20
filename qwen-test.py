'''
Testing script that performs testing/inference on the fine-tuned model
Inputs:
    1. test.jsonl file (contains paths in NRP to test images & label txt files)
    2. Fine-tuned model folder path
Output:
    1. Path to a folder that contains .txt files of the model's predictions
    Model predictions shall be txt files containing label(s) & bounding box coordinate(s)
    Ex: Paper-D 60 120 444 518
        Paper-A 292 559 562 810

'''
import os
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLProcessor
from unsloth import FastVisionModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH  = ""
TEST_JSONL    = ""
OUTPUT_DIR    = ""

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model 
model, _ = FastVisionModel.from_pretrained(
    BASE_MODEL_ID,
    load_in_4bit=True,
    device_map="auto",
)
model.load_adapter(ADAPTER_PATH)
model.eval()

processor = Qwen2_5_VLProcessor.from_pretrained(BASE_MODEL_ID)

print("Model and processor loaded.")

# Dataset loading
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

samples = load_jsonl(TEST_JSONL)
print(f"Loaded {len(samples)} test samples.")

# Inference message
def run_inference(sample):
    user_msg = next(m for m in sample["messages"] if m["role"] == "user")

    image_path = None
    for item in user_msg["content"]:
        if item["type"] == "image":
            image_path = item["image"]

    if image_path is None:
        raise ValueError("No image path found in sample")

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that detects jam causing objects in images. "
                "The possible jam causing objects are: fabrics, rigid-plastic, "
                "non-recyclables, large-plastic-films, unopened-plastic-bags, "
                "metal, wrappables, wood."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Detect the waste objects in this image and output their "
                        "bounding boxes in the format: class_name xmin xmax ymin ymax"
                    )
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = processor.decode(generated_ids, skip_special_tokens=True)

    return image_path, result

# Parse model output
def parse_boxes(text):
    boxes = []
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xmin, xmax, ymin, ymax = parts
        try:
            boxes.append(f"{cls} {int(xmin)} {int(xmax)} {int(ymin)} {int(ymax)}")
        except ValueError:
            continue
    return boxes

# Testing loop
print("Running inference on test set...")

for idx, sample in enumerate(samples):
    try:
        image_path, output_text = run_inference(sample)
        boxes = parse_boxes(output_text)

        image_name = os.path.basename(image_path)
        base_name, _ = os.path.splitext(image_name)
        pred_file = os.path.join(
            OUTPUT_DIR, f"{base_name}-pred.txt"
        )

        with open(pred_file, "w") as f:
            for line in boxes:
                f.write(line + "\n")

        print(f"[{idx+1}/{len(samples)}] Wrote {pred_file}")

    except Exception as e:
        print(f"[{idx+1}] ERROR:", e)

print("Inference complete.")

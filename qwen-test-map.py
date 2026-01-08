import json
import os
import traceback
import numpy as np
import torch
from PIL import Image

from transformers import Qwen2_5_VLProcessor
from unsloth import FastVisionModel

from mean_average_precision import MetricBuilder


MODEL_PATH  = "/workspace/models/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR  = "/workspace/output/qwen_unsloth"
DATASET_DIR = "/workspace/data/jam-causing-material"
TEST_JSONL  = "/workspace/data/test.jsonl"


CLASSES = [
    "fabrics", "rigid-plastic", "non-recyclables", "large-plastic-films",
    "unopened-plastic-bags", "metal", "wrappables", "wood"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def extract_image_path(sample):
    user_msg = next(m for m in sample["messages"] if m["role"] == "user")
    image_path = None
    for item in user_msg["content"]:
        if isinstance(item, dict) and item.get("type") == "image":
            image_path = item.get("image")
    if image_path is None:
        raise ValueError("No image path found in sample user message")
    return image_path


def run_inference(model, processor, sample):
    image_path = extract_image_path(sample)
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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = processor.decode(generated_ids, skip_special_tokens=True)
    return result


def parse_boxes(text):
    boxes = []
    lines = text.strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, *coords = parts
        try:
            xmin, xmax, ymin, ymax = map(int, coords)
            if xmin > xmax or ymin > ymax:
                continue
            boxes.append((cls, xmin, xmax, ymin, ymax))
        except ValueError:
            continue
    return boxes


def parse_ground_truth(sample):
    try:
        gt_msg = next(m for m in sample["messages"] if m["role"] == "assistant")
        return parse_boxes(gt_msg["content"])
    except StopIteration:
        return []


def parse_for_map_from_boxes(boxes, default_score=1.0):
    preds = []
    for cls, xmin, xmax, ymin, ymax in boxes:
        if cls not in CLASS_TO_IDX:
            continue
        x1, y1, x2, y2 = xmin, ymin, xmax, ymax
        preds.append([x1, y1, x2, y2, float(default_score), CLASS_TO_IDX[cls]])
    return np.array(preds) if preds else np.empty((0, 6), dtype=float)


def parse_gt_for_map_from_boxes(boxes):
    gts = []
    for cls, xmin, xmax, ymin, ymax in boxes:
        if cls not in CLASS_TO_IDX:
            continue
        x1, y1, x2, y2 = xmin, ymin, xmax, ymax
        gts.append([x1, y1, x2, y2, CLASS_TO_IDX[cls]])

    if not gts:
        return np.empty((0, 7), dtype=float)

    gts = np.array(gts, dtype=float)
    difficult = np.zeros((len(gts), 1), dtype=float)
    crowd = np.zeros((len(gts), 1), dtype=float)
    return np.concatenate([gts, difficult, crowd], axis=1)


def compute_map_for_single_image(detections, ground_truths):
    metric = MetricBuilder.build_evaluation_metric(
        "map_2d",
        async_mode=False,
        num_classes=len(CLASSES),
    )
    metric.add(detections, ground_truths)

    res_50 = metric.value(iou_thresholds=[0.5])["mAP"]
    iou_range = np.arange(0.5, 1.0, 0.05)
    res_5095 = metric.value(iou_thresholds=iou_range)["mAP"]
    return float(res_50), float(res_5095)


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:False"

    try:
        _ = MetricBuilder
    except Exception as e:
        raise RuntimeError(
            "mean-average-precision is not installed in this environment. "
            "Install it in your Docker image (pip install mean-average-precision)."
        ) from e

    print("[INFO] Loading model...")
    model, _ = FastVisionModel.from_pretrained(
        MODEL_PATH,
        load_in_4bit=True,
        device_map="auto",
    )
    model.load_adapter(OUTPUT_DIR)
    model.eval()

    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_PATH)
    print("[INFO] Model loaded.")

    print("[INFO] Loading test dataset...")
    samples = load_jsonl(TEST_JSONL)
    print(f"[INFO] Loaded {len(samples)} samples.")

    global_metric = MetricBuilder.build_evaluation_metric(
        "map_2d",
        async_mode=False,
        num_classes=len(CLASSES),
    )

    valid_count = 0

    print("\n============================================================")
    print("Running Inference + Per-Image mAP@50 and mAP@50-95")
    print("============================================================")

    for idx, sample in enumerate(samples, start=1):
        try:
            pred_text = run_inference(model, processor, sample)

            pred_boxes = parse_boxes(pred_text)
            gt_boxes = parse_ground_truth(sample)

            detections = parse_for_map_from_boxes(pred_boxes, default_score=1.0)
            ground_truths = parse_gt_for_map_from_boxes(gt_boxes)

            if detections.shape[0] == 0 and ground_truths.shape[0] == 0:
                print(f"Image {idx:03d} | skipped (no preds, no GT)")
                continue

            if ground_truths.shape[0] == 0:
                print(f"Image {idx:03d} | skipped (no GT)")
                continue

            map50_i, map5095_i = compute_map_for_single_image(detections, ground_truths)
            print(f"Image {idx:03d} | mAP@50: {map50_i:.4f} | mAP@50-95: {map5095_i:.4f}")

            global_metric.add(detections, ground_truths)
            valid_count += 1

        except Exception as e:
            print(f"Image {idx:03d} | ERROR: {e}")
            traceback.print_exc()
            continue

    print("\n============================================================")
    print("FINAL mAP RESULTS ")
    print("\n============================================================")

    if valid_count == 0:
        print("[WARN] No valid samples were evaluated (valid_count=0).")
        return

    result_50 = global_metric.value(iou_thresholds=[0.5])
    map50 = result_50["mAP"]

    iou_range = np.arange(0.5, 1.0, 0.05)
    result_all = global_metric.value(iou_thresholds=iou_range)
    map5095 = result_all["mAP"]

    print(f"Evaluated images: {valid_count}/{len(samples)}")
    print(f"mAP@50   : {map50:.4f}")
    print(f"mAP@50-95: {map5095:.4f}")


if __name__ == "__main__":
    main()

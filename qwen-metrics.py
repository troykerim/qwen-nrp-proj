import os
import random
from collections import Counter
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.detection.mean_ap import MeanAveragePrecision


BOX_FORMAT = "XXYY"  # xmin xmax ymin ymax 

PRED_SUFFIX = "-pred.txt"

GT_LABEL_DIR = "/home/troy/jam-causing-material/test/labels"
PRED_LABEL_DIR = "/home/troy/jam-causing-material-predictions"
GT_IMAGE_DIR = "/home/troy/jam-causing-material-V2/test/images"

NUM_DEBUG_IMAGES = 3


def normalize_pred_name(filename):
    if filename.endswith(PRED_SUFFIX):
        return filename[:-len(PRED_SUFFIX)] + ".txt"
    return filename


# LABEL PARSING 
def parse_label_file(path):
    labels = []
    boxes = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise RuntimeError(f"Invalid label line in {path}: {line}")

            cls = parts[0]
            xmin, xmax, ymin, ymax = map(float, parts[1:])

            if xmax <= xmin or ymax <= ymin:
                continue

            labels.append(cls)
            boxes.append((xmin, xmax, ymin, ymax))

    return labels, boxes


# CONFUSION MATRIX 
def update_confusion_matrix(cm, class_to_idx, gt_labels, pred_labels):
    BG = len(class_to_idx)

    gt_counts = Counter(gt_labels)
    pred_counts = Counter(pred_labels)

    for cls in class_to_idx:
        m = min(gt_counts.get(cls, 0), pred_counts.get(cls, 0))
        if m > 0:
            i = class_to_idx[cls]
            cm[i, i] += m
            gt_counts[cls] -= m
            pred_counts[cls] -= m

    gt_left = []
    pred_left = []

    for cls, cnt in gt_counts.items():
        gt_left.extend([cls] * cnt)
    for cls, cnt in pred_counts.items():
        pred_left.extend([cls] * cnt)

    gt_left.sort()
    pred_left.sort()

    k = min(len(gt_left), len(pred_left))
    for i in range(k):
        gi = class_to_idx[gt_left[i]]
        pi = class_to_idx[pred_left[i]]
        cm[gi, pi] += 1

    for cls in pred_left[k:]:
        pi = class_to_idx[cls]
        cm[BG, pi] += 1

    for cls in gt_left[k:]:
        gi = class_to_idx[cls]
        cm[gi, BG] += 1



def draw_boxes(image, labels, boxes, color):
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    for cls, (xmin, xmax, ymin, ymax) in zip(labels, boxes):
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            outline=color,
            width=4
        )
        draw.text(
            (xmin + 4, ymin + 4),
            cls,
            fill=color,
            font=font
        )

    return image


def visualize_pascal_voc_samples(gt_map, pred_map):
    common_files = sorted(gt_map.keys())

    if len(common_files) <= NUM_DEBUG_IMAGES:
        samples = common_files
    else:
        samples = random.sample(common_files, NUM_DEBUG_IMAGES)

    for name in samples:
        img_name = os.path.splitext(name)[0] + ".jpg"
        img_path = os.path.join(GT_IMAGE_DIR, img_name)

        if not os.path.exists(img_path):
            continue

        gt_labels, gt_boxes = parse_label_file(gt_map[name])
        pred_labels, pred_boxes = parse_label_file(pred_map[name])

        img_gt = Image.open(img_path).convert("RGB")
        img_pred = img_gt.copy()

        img_gt = draw_boxes(img_gt, gt_labels, gt_boxes, "green")
        img_pred = draw_boxes(img_pred, pred_labels, pred_boxes, "red")

        fig, axs = plt.subplots(1, 2, figsize=(18, 9))
        axs[0].imshow(img_gt)
        axs[0].set_title("Original Ground Truth", fontsize=16)
        axs[1].imshow(img_pred)
        axs[1].set_title("Model's Prediction", fontsize=16)

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.show()



def evaluate():

    gt_files = sorted(f for f in os.listdir(GT_LABEL_DIR) if f.endswith(".txt"))
    pred_files = sorted(f for f in os.listdir(PRED_LABEL_DIR) if f.endswith(".txt"))

    gt_map = {f: os.path.join(GT_LABEL_DIR, f) for f in gt_files}
    pred_map = {
        normalize_pred_name(f): os.path.join(PRED_LABEL_DIR, f)
        for f in pred_files
    }

    if set(gt_map) != set(pred_map):
        raise RuntimeError("Filename mismatch between GT and predictions")

    # Random visual sanity check (every run is different)
    visualize_pascal_voc_samples(gt_map, pred_map)

    classes = sorted({
        line.split()[0]
        for p in gt_map.values()
        for line in open(p)
        if line.strip()
    })

    class_to_idx = {c: i for i, c in enumerate(classes)}

    metric_map = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )

    total_tp = total_fp = total_fn = 0

    for name in gt_map:
        gt_labels, gt_boxes = parse_label_file(gt_map[name])
        pred_labels, pred_boxes = parse_label_file(pred_map[name])

        gt_counts = Counter(gt_labels)
        pred_counts = Counter(pred_labels)

        for cls in classes:
            t = min(gt_counts.get(cls, 0), pred_counts.get(cls, 0))
            total_tp += t
            total_fp += max(0, pred_counts.get(cls, 0) - gt_counts.get(cls, 0))
            total_fn += max(0, gt_counts.get(cls, 0) - pred_counts.get(cls, 0))

        gt_ids = torch.tensor([class_to_idx[c] for c in gt_labels], dtype=torch.long)
        pred_ids = torch.tensor([class_to_idx[c] for c in pred_labels], dtype=torch.long)
        scores = torch.ones(len(pred_ids))

        metric_map.update(
            preds=[{"boxes": torch.tensor(pred_boxes), "scores": scores, "labels": pred_ids}],
            target=[{"boxes": torch.tensor(gt_boxes), "labels": gt_ids}],
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    map_results = metric_map.compute()

    print("\n===== Detection Metrics =====\n")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"mAP@50:     {map_results['map_50']:.4f}")
    print(f"mAP@50–95:  {map_results['map']:.4f}")


if __name__ == "__main__":
    evaluate()

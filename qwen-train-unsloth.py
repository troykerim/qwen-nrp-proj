import os
import json
from pathlib import Path

import torch
import wandb
from PIL import Image

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import AutoProcessor, EarlyStoppingCallback, TrainerCallback

MODEL_PATH = "/workspace/models/Qwen2.5-VL-7B-Instruct"
DATASET_DIR = "/workspace/data/jam-causing-voc-aug-voc"
TRAIN_JSONL = "/workspace/data/train.jsonl"
VALID_JSONL = "/workspace/data/valid.jsonl"
OUTPUT_DIR = "/workspace/output/qwen_unsloth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Path check:")
print("Model:", os.path.exists(MODEL_PATH))
print("Dataset:", os.path.exists(DATASET_DIR))
print("Train JSONL:", os.path.exists(TRAIN_JSONL))
print("Valid JSONL:", os.path.exists(VALID_JSONL))
print()

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def normalize_image_path(img_path: str) -> str:
    p = Path(img_path)

    # Case 1: relative path from JSONL
    if not p.is_absolute():
        return str(Path(DATASET_DIR) / img_path)

    # Case 2: absolute path already exists in container
    if p.exists():
        return str(p)

    # Case 3: absolute path from local PC; remap by basename into dataset
    candidate = Path(DATASET_DIR) / p.name
    if candidate.exists():
        return str(candidate)

    # Case 4: try common train/valid/test image folder reconstruction
    parts = p.parts
    if "images" in parts:
        idx = parts.index("images")
        tail = parts[idx + 1 :]
        candidate = Path(DATASET_DIR) / "images" / Path(*tail)
        if candidate.exists():
            return str(candidate)

    return str(p)


def is_valid_image(image_path: str) -> tuple[bool, str]:
    p = Path(image_path)

    if not p.exists():
        return False, f"missing image: {p}"

    if not p.is_file():
        return False, f"not a file: {p}"

    if p.stat().st_size == 0:
        return False, f"empty image file: {p}"

    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as e:
        return False, f"corrupt/unreadable image: {p} | {e}"

    return True, ""


def infer_label_path(image_path: str) -> Path:
    p = Path(image_path)

    # Common case: .../images/file.jpg -> .../labels/file.txt
    if p.parent.name == "images":
        return p.parent.parent / "labels" / f"{p.stem}.txt"

    # Fallback: same folder, txt extension
    return p.with_suffix(".txt")


def is_valid_label_for_image(image_path: str) -> tuple[bool, str]:
    label_path = infer_label_path(image_path)

    if not label_path.exists():
        return False, f"missing label: {label_path}"

    if not label_path.is_file():
        return False, f"label is not a file: {label_path}"

    if label_path.stat().st_size == 0:
        return False, f"empty label file: {label_path}"

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) == 0:
            return False, f"label has no usable lines: {label_path}"
    except Exception as e:
        return False, f"unreadable label: {label_path} | {e}"

    return True, ""


def sanitize_example(item: dict) -> tuple[dict | None, str]:
    found_image = False

    for msg in item.get("messages", []):
        if msg.get("role") != "user":
            continue

        for content in msg.get("content", []):
            if content.get("type") != "image":
                continue

            found_image = True
            original_img = content.get("image", "")
            normalized_img = normalize_image_path(original_img)
            content["image"] = normalized_img

            ok_img, reason_img = is_valid_image(normalized_img)
            if not ok_img:
                return None, reason_img

            ok_lbl, reason_lbl = is_valid_label_for_image(normalized_img)
            if not ok_lbl:
                return None, reason_lbl

    if not found_image:
        return None, "no image found in example"

    return item, ""


def load_and_filter_jsonl(path: str, split_name: str):
    data = []
    skipped = 0
    skip_reasons = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                skipped += 1
                skip_reasons["empty jsonl line"] = skip_reasons.get("empty jsonl line", 0) + 1
                continue

            try:
                item = json.loads(line)
            except Exception as e:
                skipped += 1
                key = f"bad json line: {e}"
                skip_reasons[key] = skip_reasons.get(key, 0) + 1
                continue

            sanitized_item, reason = sanitize_example(item)
            if sanitized_item is None:
                skipped += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            data.append(sanitized_item)

    print(f"{split_name} kept samples: {len(data)}")
    print(f"{split_name} skipped samples: {skipped}")

    if skip_reasons:
        print(f"{split_name} skip summary:")
        shown = 0
        for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {count} | {reason}")
            shown += 1
            if shown >= 20:
                remaining = len(skip_reasons) - shown
                if remaining > 0:
                    print(f"  ... and {remaining} more skip reason(s)")
                break
    print()

    return data


class PrettyLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        out = {}

        if "loss" in logs and logs["loss"] is not None:
            out["loss"] = f"{float(logs['loss']):.4f}"

        if "eval_loss" in logs and logs["eval_loss"] is not None:
            out["eval_loss"] = f"{float(logs['eval_loss']):.4f}"

        if "grad_norm" in logs and logs["grad_norm"] is not None:
            out["grad_norm"] = f"{float(logs['grad_norm']):.4f}"

        if "learning_rate" in logs and logs["learning_rate"] is not None:
            out["learning_rate"] = f"{float(logs['learning_rate']):.3e}"

        if "epoch" in logs and logs["epoch"] is not None:
            out["epoch"] = logs["epoch"]

        if out:
            print(out)


train_data = load_and_filter_jsonl(TRAIN_JSONL, "Train")
valid_data = load_and_filter_jsonl(VALID_JSONL, "Valid")

if len(train_data) == 0:
    raise RuntimeError("No valid training samples remain after filtering.")

if len(valid_data) == 0:
    raise RuntimeError("No valid validation samples remain after filtering.")

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    random_state=42,
)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_data,
    eval_dataset=valid_data,
    callbacks=[
        PrettyLogCallback(),
        EarlyStoppingCallback(early_stopping_patience=5),
    ],
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_steps=50,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
        report_to="wandb",
        run_name="qwen-NRP-7B-VL3",
        seed=42,
    ),
)

print("Starting training...\n")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nTraining complete. Model saved to {OUTPUT_DIR}\n")

print("\nTraining Parameters used in this run:")
print(f"LoRA rank:          {LORA_R}")
print(f"LoRA alpha:         {LORA_ALPHA}")
print(f"LoRA dropout:       {LORA_DROPOUT}")
print(f"Learning rate:      {trainer.args.learning_rate}")
print(f"Num train epochs:   {trainer.args.num_train_epochs}")
print(f"Warmup ratio:       {trainer.args.warmup_ratio}")
print(f"Optimizer:          {trainer.args.optim}")

if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    used_vram = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

    print("\nGPU VRAM Summary:")
    print(f"Total GPU VRAM available: {total_vram:.2f} GB")
    print(f"Peak GPU VRAM used:       {used_vram:.2f} GB")
else:
    print("\nGPU VRAM Summary:")
    print("CUDA not available")
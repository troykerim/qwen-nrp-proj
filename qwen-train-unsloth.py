import os

# Disable TorchInductor / Triton compilation (REQUIRED for Nautilus)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

# Memory safety
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import AutoProcessor

# Nautilus paths (UNCHANGED)
MODEL_PATH    = "/workspace/models/Qwen2.5-VL-7B-Instruct"
DATASET_DIR   = "/workspace/data/dataset"
TRAIN_JSONL   = "/workspace/data/train.jsonl"
VALID_JSONL   = "/workspace/data/valid.jsonl"
OUTPUT_DIR    = "/workspace/output/qwen_unsloth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Path check:")
print("Model:", os.path.exists(MODEL_PATH))
print("Dataset:", os.path.exists(DATASET_DIR))
print("Train JSONL:", os.path.exists(TRAIN_JSONL))
print("Valid JSONL:", os.path.exists(VALID_JSONL))

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)

            # Fix relative image paths using DATASET_DIR
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    for content in msg.get("content", []):
                        if content.get("type") == "image":
                            img = content["image"]
                            if not img.startswith("/"):
                                content["image"] = os.path.join(DATASET_DIR, img)

            data.append(item)
    return data

train_data = load_jsonl(TRAIN_JSONL)
valid_data = load_jsonl(VALID_JSONL)

print(f"Train samples: {len(train_data)}")
print(f"Valid samples: {len(valid_data)}")

# Load model with Unsloth
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
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
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
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        optim="adamw_8bit",
        weight_decay=0.01,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
        report_to="none",
        seed=42,
    ),
)

print("Starting training...\n")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training complete. Model snaved to {OUTPUT_DIR}")

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print("FINAL GPU MEMORY STATS")
    print(f"GPU: {gpu.name}")
    print(f"Max reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
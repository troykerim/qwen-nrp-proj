import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from qwen_vl_utils import process_vision_info
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from transformers import AutoProcessor

MODEL_PATH    = "/workspace/models/Qwen2.5-VL-7B-Instruct"
DATASET_DIR   = "/workspace/data/dataset"
TRAINING_DATA = "/workspace/data/train.jsonl"
VAL_DATA      = "/workspace/data/valid.jsonl"
OUTPUT_DIR    = "/workspace/output/qwen_unsloth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Path check:")
print("Model:", os.path.exists(MODEL_PATH))
print("Dataset:", os.path.exists(DATASET_DIR))
print("Train JSONL:", os.path.exists(TRAINING_DATA))
print("Valid JSONL:", os.path.exists(VAL_DATA))

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
    r=16,
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

def load_jsonl_with_images(jsonl_path, dataset_dir):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            for msg in item["messages"]:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content["type"] == "image":
                            img = content["image"]
                            if not img.startswith("/"):
                                content["image"] = os.path.join(dataset_dir, img)
            data.append(item)
    return data

train_dataset = load_jsonl_with_images(TRAINING_DATA, DATASET_DIR)
val_dataset   = load_jsonl_with_images(VAL_DATA, DATASET_DIR)

print(f"Train samples: {len(train_dataset)}")
print(f"Valid samples: {len(val_dataset)}")

class WasteDetectionUnslothCollator:
    def __init__(self, processor, max_length=2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features):
        texts, batch_images, batch_videos = [], [], []

        for f in features:
            msgs = f["messages"]
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            imgs, vids = process_vision_info(msgs)
            batch_images.append(imgs or [])
            batch_videos.append(vids or [])

        images = None if all(len(x) == 0 for x in batch_images) else batch_images
        videos = None if all(len(x) == 0 for x in batch_videos) else batch_videos

        batch = self.processor(
            text=texts,
            images=images,
            videos=videos,
            max_length=self.max_length,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        for idx, f in enumerate(features):
            labels[idx] = -100
            assistant_msg = None
            for msg in f["messages"]:
                if msg["role"] == "assistant":
                    assistant_msg = msg["content"]
                    break

            if assistant_msg is None:
                continue

            asst_tokens = self.processor.tokenizer.encode(
                assistant_msg, add_special_tokens=False
            )

            seq_len = batch["attention_mask"][idx].sum().item()
            asst_len = len(asst_tokens)

            if 0 < asst_len < seq_len:
                start = seq_len - asst_len
                labels[idx, start:seq_len] = batch["input_ids"][idx, start:seq_len]

        batch["labels"] = labels
        return batch

# Training Parameters
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=WasteDetectionUnslothCollator(processor),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=3e-4,
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

print(f"Training complete. Model saved to {OUTPUT_DIR}")

# MEMORY STATS
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print("\n" + "="*70)
    print("FINAL GPU MEMORY STATS (UNSLOTH)")
    print("="*70)
    print(f"GPU Name:        {gpu.name}")
    print(f"Total VRAM:     {gpu.total_memory / 1024**3:.2f} GB")
    print(f"Max Reserved:   {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    print(f"Max Allocated:  {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("="*70)

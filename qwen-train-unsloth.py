import os
import json
import torch
import wandb

# Nautilus stability flags
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import AutoProcessor

MODEL_PATH  = "/workspace/models/Qwen2.5-VL-7B-Instruct"
DATASET_DIR = "/workspace/data/jam-causing-material"
TRAIN_JSONL = "/workspace/data/train.jsonl"
VALID_JSONL = "/workspace/data/valid.jsonl"

OUTPUT_DIR  = "/workspace/output/qwen_unsloth3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Path check:")
print("Model:", os.path.exists(MODEL_PATH))
print("Dataset:", os.path.exists(DATASET_DIR))
print("Train JSONL:", os.path.exists(TRAIN_JSONL))
print("Valid JSONL:", os.path.exists(VALID_JSONL))
print()

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
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
print(f"Valid samples: {len(valid_data)}\n")

# LoRA Parameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# W&B configs
# Delete this
'''wandb.init(
    project=os.environ["WANDB_PROJECT"],
    entity="troy-kerim-26",
    reinit=True,
    config={
        "model": "Qwen2.5-VL-7B-Instruct",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "learning_rate": 3e-4,
        "epochs": 3,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "optimizer": "adamw_8bit",
    },
)'''

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
        report_to="wandb",  # keep this
        run_name='qwen-NRP-7B-VL', # Rename if I like.
        seed=42,
    ),
)

print("Starting training...\n")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nTraining complete. Model saved to {OUTPUT_DIR}\n")



# W&B logging, 
'''if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    wandb.log({
        "gpu/name": gpu.name,
        "gpu/max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "gpu/max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    })

wandb.finish()'''
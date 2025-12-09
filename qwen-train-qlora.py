import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from transformers import Trainer
from collections import defaultdict
import json

MODEL_PATH = "/workspace/models/Qwen2.5-VL-7B-Instruct"
DATASET_DIR = "/workspace/data/dataset"
TRAINING_DATA = "/workspace/data/train.jsonl"
VAL_DATA = "/workspace/data/valid.jsonl"
OUTPUT_DIR = "/workspace/output/qwen_qlora2"  # Original was "/workspace/output/qwen_qlora"

class WasteDetectionDataset(torch.utils.data.Dataset):
    """Custom dataset that doesn't use Arrow/Datasets library"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"messages": self.data[idx]["messages"]}

class WasteDetectionDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        texts, batch_images, batch_videos = [], [], []

        for f in features:
            msgs = f["messages"]
            # Apply chat template
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
            max_length=2048,   #None, It was either 512 or was too large?
            videos=videos,
            padding=True, # padding=True, truncation=True,
            truncation = False,
            return_tensors="pt"
        )

        # Get labels from input_ids
        labels = batch["input_ids"].clone()

        # For Qwen models, we need to mask everything except assistant response
        # The key is to find where assistant response tokens start
        for idx in range(len(features)):
            # Get the full sequence
            input_ids_list = batch["input_ids"][idx].tolist()

            # Mask everything first
            labels[idx] = -100

            # Find assistant content from original messages
            msgs = features[idx]["messages"]
            assistant_msg = None
            for msg in msgs:
                if msg["role"] == "assistant":
                    assistant_msg = msg["content"]
                    break

            if assistant_msg:
                # Tokenize just the assistant response
                asst_tokens = self.processor.tokenizer.encode(
                    assistant_msg,
                    add_special_tokens=False
                )

                # Find these tokens in the full sequence (from the end)
                seq_len = (batch["attention_mask"][idx] == 1).sum().item()

                # Match from the end backwards
                asst_len = len(asst_tokens)
                if asst_len > 0 and asst_len < seq_len:
                    # Check if the tokens match at the end
                    expected_start = seq_len - asst_len

                    # Unmask the assistant portion
                    labels[idx, expected_start:seq_len] = batch["input_ids"][idx, expected_start:seq_len]

        batch["labels"] = labels
        return batch

print("Collator defined - matching from end of sequence")

# QLoRA 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model with 8-bit quantization
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

# Load processor
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)
print(f"Model loaded with 8-bit quantization: {type(model).__name__}")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],     # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing
model.enable_input_require_grads()
model.config.use_cache = False

print("8-bit QLoRA configured")

import json
# Load training data
print("Loading training data...")
train_data = []
with open(TRAINING_DATA, 'r') as f:
    for line in f:
        train_data.append(json.loads(line))

train_dataset = WasteDetectionDataset(train_data)
print(f"Loaded {len(train_dataset)} training samples")

# Load validation data
print("Loading validation data...")
val_data = []
# with open(VALIDATION_DATA, 'r') as f:
with open(VAL_DATA, 'r') as f:
    for line in f:
        val_data.append(json.loads(line))

val_dataset = WasteDetectionDataset(val_data)
print(f"Loaded {len(val_dataset)} validation samples")

print(f"\n{'='*70}")
print(f"Dataset Summary:")
print(f"  Train:      {len(train_dataset)} examples")
print(f"  Validation: {len(val_dataset)} examples")
print(f"  Total:      {len(train_dataset) + len(val_dataset)} examples")
print(f"{'='*70}")

print("Configuring TrainingArguments...")

# Training Parameters
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,  # originally 5

    # Sizes for QLoRA
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    # Learning rate
    learning_rate=1.5e-4,     # 2e-4
    warmup_steps=50,         # 100, increments of 50

    # Evaluation
    eval_strategy="steps",
    eval_steps=50,

    # Logging
    logging_steps=10,
    logging_dir=f"{OUTPUT_DIR}/logs",

    # Saving
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Optimization
    gradient_checkpointing=False,   # Was original set to true, but was sending a Warning in the last cell
    optim="adamw_8bit",     
    # optim="adamw_torch",  # Should be ok with QLoRA?
    max_grad_norm=0.3,    # Gradient clipping to prevent exploding gradients

    # Memory
    dataloader_pin_memory=True,
    remove_unused_columns=False,

    # Precision
    bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),
    fp16=not (torch.cuda.get_device_capability()[0] >= 8),

    # Reporting
    report_to="none",
)

print("TrainingArguments ready.\n")

print("Initializing Trainer...")

data_collator = WasteDetectionDataCollator(processor)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
print("Trainer initialized.\n")

print("="*70)
print("FINAL COLLATOR VERIFICATION")
print("="*70)

# Test two samples
test_batch = data_collator([train_dataset[0], train_dataset[1]])

print(f"\nBatch size: {test_batch['input_ids'].shape[0]}")
print(f"Sequence length: {test_batch['input_ids'].shape[1]}")

for i in range(test_batch['input_ids'].shape[0]):
    total = test_batch["labels"][i].numel()
    non_masked = (test_batch["labels"][i] != -100).sum().item()

    msgs = [train_dataset[0], train_dataset[1]][i]["messages"]
    expected = 0
    for msg in msgs:
        if msg["role"] == "assistant":
            expected = len(processor.tokenizer.encode(
                msg["content"], add_special_tokens=False
            ))
            break

    print(f"\nExample {i+1}:")
    print(f"  Learning from: {non_masked}/{total} tokens ({non_masked/total*100:.1f}%)")
    print(f"  Expected: {expected} tokens")
    print(f"  Match: {'PERFECT' if abs(expected - non_masked) < 5 else 'MISMATCH'}")

print("\n" + "="*70)
print("COLLATOR STATUS: READY FOR TRAINING ✓")
print("="*70 + "\n")
print(f"Training on {len(train_dataset)} examples")
print(f"Validating on {len(val_dataset)} examples")
print("Starting training...\n")

trainer.train()
print("\nSaving model and processor...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"Training complete! Model saved to {OUTPUT_DIR}")

# REVISE
by_step = defaultdict(dict)
for log in trainer.state.log_history:
    step = log.get("step")
    if step is None:
        continue
    for k, v in log.items():
        if k == "step":
            continue
        by_step[step][k] = v

print("Step\tTrain Loss\tEval Loss")
for step in sorted(by_step.keys()):
    train_loss = by_step[step].get("loss")
    eval_loss = by_step[step].get("eval_loss")
    if train_loss is None and eval_loss is None:
        continue
    print(f"{step}\t{train_loss}\t{eval_loss}")
import os
import json
import torch

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
from torch.utils.data import DataLoader

MODEL_PATH  = "/workspace/models/Qwen2.5-VL-7B-Instruct"
DATASET_DIR = "/workspace/data/jam-causing-material"
TRAIN_JSONL = "/workspace/data/train.jsonl"
VALID_JSONL = "/workspace/data/valid.jsonl"
TEST_JSONL  = "/workspace/data/test.jsonl"

OUTPUT_DIR  = "/workspace/output/qwen_unsloth2"  # Change output directory for each training session
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Path check:")
print("Model:", os.path.exists(MODEL_PATH))
print("Dataset:", os.path.exists(DATASET_DIR))
print("Train JSONL:", os.path.exists(TRAIN_JSONL))
print("Valid JSONL:", os.path.exists(VALID_JSONL))
print("Test JSONL:", os.path.exists(TEST_JSONL))
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
test_data  = load_jsonl(TEST_JSONL)

print(f"Train samples: {len(train_data)}")
print(f"Valid samples: {len(valid_data)}")
print(f"Test samples : {len(test_data)}\n")

# LoRA Parameters 
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

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
        learning_rate=1.5e-4,  # 2e-4
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

print(f"\nTraining complete. Model saved to {OUTPUT_DIR}\n")
# End of training loop

# Testing loop (below), may need to remove it because of NRP?
def extract_assistant_text(text):
    if "<|im_start|>assistant" in text:
        return text.split("<|im_start|>assistant")[-1].strip()
    return text.strip()

test_collator = UnslothVisionDataCollator(model, tokenizer)

class TestDataset:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

test_dataset = TestDataset(test_data)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=test_collator,
)

model.eval()
results = []

print("Running evaluation on test set...")

with torch.no_grad():
    for idx, batch in enumerate(test_dataloader):
        input_ids = batch["input_ids"].to(model.device)
        pixel_values = batch["pixel_values"].to(model.device)
        image_grid_thw = batch["image_grid_thw"].to(model.device)

        outputs = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=256,
            do_sample=False,
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = extract_assistant_text(decoded)

        gt_answer = ""
        for m in test_data[idx]["messages"]:
            if m["role"] == "assistant":
                gt_answer = m["content"]
                break

        results.append({
            "index": idx,
            "prediction": prediction,
            "ground_truth": gt_answer,
        })

        if idx < 5:
            print(f"\nSample {idx}")
            print("PREDICTION:")
            print(prediction)
            print("GROUND TRUTH:")
            print(gt_answer)

results_path = os.path.join(OUTPUT_DIR, "test_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nTest results saved to {results_path}")

# Display all hyperparameters used during this training session
args = trainer.args
print("LoRA Parameters used in this session")
print(f"  Lora Rank (r)           : {LORA_R}")
print(f"  Lora Alpha              : {LORA_ALPHA}")
print(f"  Lora Droput             : {LORA_DROPOUT}")

print("\nTraining Parameters used in this session")
print(f"  Learning Rate               : {args.learning_rate}")
print(f"  Optimizer                   : {args.optim}")
print(f"  Weight Decay                : {args.weight_decay}")
print(f"  Number of training epochs   : {args.num_train_epochs}")
print(f"  Device Train Batch Size     : {args.per_device_train_batch_size}")
print(f"  Gradient Accumulation Steps : {args.gradient_accumulation_steps}")
print(f"  Warmup Steps                : {args.warmup_steps}")
print(f"  Logging Steps               : {args.logging_steps}")
print(f"  Save Steps                  : {args.save_steps}")
print(f"  Save Total Limit            : {args.save_total_limit}")

# Check how much VRAM was used by Unsloth and by the current training session overall
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print("\nFINAL GPU MEMORY STATS")
    print(f"GPU used: {gpu.name}")
    print(f"Max reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


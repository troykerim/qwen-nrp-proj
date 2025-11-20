import torch
import json
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# Paths inside Nautilus PVC
MODEL_PATH = "/workspace/models/Qwen2.5-VL-7B-Instruct"
TRAINING_DATA = "/workspace/data/train.jsonl"
VALIDATION_DATA = "/workspace/data/valid.jsonl"
OUTPUT_DIR = "/workspace/output/qwen_qlora"


class WasteDetectionDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"messages": self.data[idx]["messages"]}
# Custom collator (VL support)
class WasteDetectionDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        texts, images_batch, videos_batch = [], [], []

        for f in features:
            msgs = f["messages"]

            # Apply chat template
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            imgs, vids = process_vision_info(msgs)
            images_batch.append(imgs or [])
            videos_batch.append(vids or [])

        images = None if all(len(x) == 0 for x in images_batch) else images_batch
        videos = None if all(len(x) == 0 for x in videos_batch) else videos_batch

        batch = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt"
        )

        labels = batch["input_ids"].clone()

        # Mask non-assistant tokens
        for i in range(len(features)):
            labels[i] = -100

            msgs = features[i]["messages"]
            assistant_msg = None
            for m in msgs:
                if m["role"] == "assistant":
                    assistant_msg = m["content"]
                    break

            if assistant_msg:
                asst_tokens = self.processor.tokenizer.encode(
                    assistant_msg, add_special_tokens=False
                )
                seq_len = (batch["attention_mask"][i] == 1).sum().item()
                asst_len = len(asst_tokens)

                if asst_len > 0 and asst_len < seq_len:
                    start = seq_len - asst_len
                    labels[i, start:seq_len] = batch["input_ids"][i, start:seq_len]

        batch["labels"] = labels
        return batch

def main():
    print("======== QLoRA Finetuning (4-bit) for Qwen2.5-VL-7B-Instruct ========")

    # Load JSONL datasets
    print("Loading training JSONL…")
    train_data = [json.loads(l) for l in open(TRAINING_DATA)]
    val_data = [json.loads(l) for l in open(VALIDATION_DATA)]

    train_dataset = WasteDetectionDataset(train_data)
    val_dataset = WasteDetectionDataset(val_data)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # 4-bit quantization config to enable QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # Load model
    print("Loading Qwen2.5-VL-7B-Instruct:.......")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)

    # Define LoRA adapters
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)

    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        bf16=True,
    )

    data_collator = WasteDetectionDataCollator(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting training…")
    trainer.train()

    print("Saving adapter…")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("\nTraining complete. Output saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

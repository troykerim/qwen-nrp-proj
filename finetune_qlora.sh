#!/bin/bash
set -e
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"


# QLoRA fine-tuning, Qwen2.5-VL-7B-Instruct
torchrun --nproc_per_node=1 src/train/train.py \
  --deepspeed scripts/zero3_offload.json \
  --model_id /workspace/models/Qwen2.5-VL-7B-Instruct \
  --data_path /workspace/data/qwen_formatted.json \
  --image_folder /workspace/data/images \
  --output_dir /workspace/output/qwen25_qlora \
  --lora_enable true \                     # keep adapter logic on
  --bits 4 \                               # quantize base model to 4-bit (QLoRA)
  --bnb_4bit_compute_dtype bfloat16 \      # compute dtype for quantized layers
  --bnb_4bit_use_double_quant true \       # double quantization for lower VRAM
  --bnb_4bit_quant_type nf4 \              # NF4 quantization (standard for QLoRA)
  --use_liger false \                      # must disable liger kernel
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --freeze_vision_tower false \
  --freeze_llm false \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --report_to none \
  --logging_steps 10 \
  --dataloader_num_workers 4 \
  --fp16 true

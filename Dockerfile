# Base image: NVIDIA PyTorch (CUDA 12.3+ compatible)
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Maintainer info (optional)
LABEL maintainer="troyk500" \
      description="Qwen2.5-VL-7B QLoRA fine-tuning environment for Nautilus"

# Create working directory
WORKDIR /workspace

# Copy Qwen-VL-Series-Finetune repo into the image
# COPY Qwen-VL-Series-Finetune /workspace/Qwen-VL-Series-Finetune

# Install dependencies
RUN pip install -U pip \
 && curl -sSL https://raw.githubusercontent.com/2U1/Qwen-VL-Series-Finetune/master/requirements.txt \
    | sed 's/contourpy==1.3.3/contourpy==1.3.2/' \
    | sed 's/ipython==9.2.0/ipython==8.28.0/' \
    | sed 's/scipy==1.16.2/scipy==1.13.1/' > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt -f https://download.pytorch.org/whl/cu128 \
 && pip install qwen-vl-utils bitsandbytes==0.43.1 peft==0.10.0 \
 && pip install flash-attn --no-build-isolation
 
# Set working directory to repo
WORKDIR /workspace/Qwen-VL-Series-Finetune

# Default command
CMD ["bash"]

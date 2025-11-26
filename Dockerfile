FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y git wget curl python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
    "transformers==4.57.1" \
    "accelerate>=0.34.2" \
    "peft>=0.13.2" \
    "bitsandbytes>=0.43.0" \
    "sentencepiece" \
    "einops" \
    "timm" \
    "qwen-vl-utils"

# Optional (May or may not worK?)
# RUN pip install "datasets" "huggingface_hub" "scipy"

ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /workspace
COPY . /workspace
CMD ["bash"]

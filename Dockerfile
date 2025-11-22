# NVIDIA PyTorch base image (CUDA 12.2 runtime) – compatible with bitsandbytes
FROM nvcr.io/nvidia/pytorch:24.03-py3

LABEL maintainer="troyk500" \
      description="Qwen2.5-VL-7B QLoRA fine-tuning environment based on InternVL Dockerfile"

# Main working directory
WORKDIR /workspace

# Install pinned dependencies 
RUN pip install -U pip \
 && pip install "numpy<2.0" \
 && pip install "scipy<1.14" \
 && pip install "contourpy<1.3.0" \
 && pip install "ipython<9.0.0"

# Install Qwen-VL utilities and training dependencies
RUN pip install qwen-vl-utils \
 && pip install bitsandbytes==0.43.1 \
 && pip install peft==0.10.0

RUN pip install flash-attn --no-build-isolation

# Install HuggingFace Transformers + Datasets
RUN pip install transformers==4.40.2 \
 && pip install datasets==2.17.1 \
 && pip install accelerate==0.28.0 \
 && pip install sentencepiece \
 && pip install pillow==10.2.0

RUN rm -rf /root/.cache/pip

CMD ["bash"]


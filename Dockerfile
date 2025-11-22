FROM nvcr.io/nvidia/pytorch:23.09-py3

LABEL maintainer="troyk500" \
      description="Qwen2.5-VL-7B QLoRA environment"
WORKDIR /workspace

# Upgrade pip first
RUN pip install --upgrade pip

# Install all deps in one go to avoid version conflicts
# Pin numpy to <2 explicitly and early in the list
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "transformers==4.40.2" \
    "datasets==2.17.1" \
    "accelerate==0.28.0" \
    "peft==0.10.0" \
    "bitsandbytes==0.43.1" \
    "sentencepiece" \
    "qwen-vl-utils" \
    "pillow==10.2.0" \
    "packaging" \
    "scipy==1.11.4" \
    && \
    pip install --no-cache-dir --no-build-isolation "flash-attn" \
    && \
    pip check  # Optional: verify no conflicts

# Clean up
RUN rm -rf /root/.cache/pip

CMD ["bash"]

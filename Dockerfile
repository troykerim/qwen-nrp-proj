# Use NVIDIA PyTorch 23.09 or 24.03
FROM nvcr.io/nvidia/pytorch:23.09-py3

LABEL maintainer="troyk500" \
      description="Clean minimal Qwen2.5-VL-7B QLoRA environment (CUDA 12.1)"
WORKDIR /workspace

RUN pip install --upgrade pip \
 && pip install packaging

RUN pip install --upgrade --force-reinstall "numpy==1.26.4"

# transformers >= 4.39 REQUIRED for AutoModelForImageTextToText
RUN pip install --upgrade --force-reinstall transformers==4.40.2

RUN pip install datasets==2.17.1 \
 && pip install accelerate==0.28.0

RUN pip install qwen-vl-utils
RUN pip install sentencepiece
RUN pip install pillow==10.2.0
RUN pip install bitsandbytes==0.43.1
RUN pip install flash-attn --no-build-isolation
RUN pip install peft==0.10.0


RUN rm -rf /root/.cache/pip

CMD ["bash"]


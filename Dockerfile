FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip && \
    mkdir -p /opt/conda/bin && \
    ln -sf /usr/bin/python3 /opt/conda/bin/python3 && \
    ln -sf /usr/bin/pip3 /opt/conda/bin/pip3 && \
    ln -sf /usr/bin/pip3 /opt/conda/bin/pip

WORKDIR /workspace

COPY requirements.base.txt /tmp/requirements.base.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.base.txt && \
    python -c "import accelerate, datasets, einops, ftfy, imageio, imageio_ffmpeg, modelscope, pandas, peft, PIL, pynvml, qwen_vl_utils, safetensors, sentencepiece, transformers; import google.protobuf; print('runtime deps ok')"

CMD ["python"]

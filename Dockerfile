FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
RUN uv venv --python 3.11
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install --no-cache-dir torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt ./
RUN uv pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py ./

RUN mkdir -p /data/huggingface /tmp/uploads

ENV HOST=0.0.0.0
ENV PORT=8000
ENV WHISPER_MODEL_ID=KBLab/kb-whisper-large
ENV WHISPER_DEVICE=cuda
ENV WHISPER_DEFAULT_LANGUAGE=sv
ENV WHISPER_MAX_UPLOAD_SIZE_MB=200
ENV WHISPER_ENABLE_DOCS=true

ENV HF_HOME=/data/huggingface
ENV HF_HUB_CACHE=/data/huggingface/hub
ENV TRANSFORMERS_CACHE=/data/huggingface/transformers
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]

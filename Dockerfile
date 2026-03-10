# ── Dockerfile ──
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ── 系统依赖（通过 deadsnakes PPA 安装 Python 3.11） ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        build-essential git git-lfs \
        libgl1-mesa-glx libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && ln -sf /usr/local/bin/pip3 /usr/local/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 先装 PyTorch（利用 Docker 缓存层） ──
RUN pip install --no-cache-dir \
        torch==2.9.0 torchvision==0.24.0

# ── 复制项目 ──
COPY . /app/

# ── 初始化并安装 SAM2 子模块 ──
RUN git submodule update --init --recursive || true \
    && if [ -f sam2/setup.py ] || [ -f sam2/pyproject.toml ]; then \
         pip install --no-cache-dir -e ./sam2; \
       fi

# ── 安装项目依赖 ──
RUN pip install --no-cache-dir \
        numpy==2.2.6 \
        opencv-python==4.12.0.88 \
        pillow==12.0.0 \
        tqdm==4.67.1 \
        datasets \
        tensorboard==2.20.0 \
        hydra-core==1.3.2 \
        transformers==4.57.1 \
        peft==0.17.1 \
        accelerate==1.11.0 \
        safetensors==0.6.2 \
        scipy==1.16.2 \
        scikit-image==0.25.2 \
        matplotlib==3.10.7 \
        pandas==2.3.3 \
        pycocotools==2.0.10 \
        timm==1.0.20 \
        tiktoken==0.12.0 \
        sentencepiece==0.2.1 \
        fvcore==0.1.5.post20221221 \
        iopath==0.1.10 \
        omegaconf==2.3.0 \
        pyyaml==6.0.3 \
        regex==2025.10.23 \
        huggingface-hub==0.35.3 \
        psutil==7.1.1 \
        gpustat==1.1.1

# ── 默认入口 ──
ENTRYPOINT ["python"]
CMD ["demo.py", "--help"]

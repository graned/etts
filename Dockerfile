# ------------------------------------------------------------
# ETTS TRAINING IMAGE  ·  Python 3.10.12 · CUDA 11.7 · cuDNN 8
# ------------------------------------------------------------
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# ── Basic system setup ───────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip espeak\
    git ffmpeg build-essential curl wget unzip \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip

# ── ETTS‑specific ENV variables (can be overridden at runtime) ─
ENV PHONEME_DICT_PATH=train/dictionaries/phoneme_dict.json \
    MANIFEST_PATH=train/manifests/etts_manifest.json \
    CHECKPOINT_DIR=train/checkpoints \
    SAMPLES_PATH=train/samples \
    TRAIN_PHONEME_STEPS=1 \
    TRAIN_PHONEME_MELS=2 \
    TRAIN_EPOCHS=100 \
    TRAIN_LR=1e-3

# ── Python deps & code ────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Create checkpoint dir if volume not mounted
RUN mkdir -p ${CHECKPOINT_DIR}

# Optional port for TensorBoard
EXPOSE 6006

CMD ["tensorBoard", "--logdir=runs"]


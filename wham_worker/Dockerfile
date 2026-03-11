FROM docker.io/pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    libpq-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    smplx \
    einops \
    yacs \
    loguru \
    tqdm \
    scikit-image \
    opencv-python-headless \
    imageio[ffmpeg] \
    timm==0.6.13 \
    progress \
    joblib \
    chumpy \
    gdown \
    celery[redis]==5.4.0 \
    redis==5.1.1 \
    sqlalchemy==2.0.36 \
    psycopg2-binary==2.9.9

# Clone WHAM (without recursive — skip ViTPose/DPVO submodules)
RUN git clone --depth 1 https://github.com/yohanshin/WHAM.git /opt/wham

ENV PYTHONPATH="/opt/wham:${PYTHONPATH}"

COPY . .

CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "-Q", "wham_3d", "-c", "1", "--prefetch-multiplier=1"]

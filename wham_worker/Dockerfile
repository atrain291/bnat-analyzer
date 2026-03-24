FROM docker.io/pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /app

# System deps + build tools for ffmpeg with NVDEC
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    make \
    nasm \
    pkg-config \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install nv-codec-headers (for NVDEC/NVENC support in ffmpeg)
RUN git clone --depth 1 -b n11.1.5.3 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
    && cd /tmp/nv-codec-headers && make install && rm -rf /tmp/nv-codec-headers

# Build ffmpeg with NVDEC hardware acceleration
RUN wget -q https://ffmpeg.org/releases/ffmpeg-6.1.2.tar.xz -O /tmp/ffmpeg.tar.xz \
    && cd /tmp && tar xf ffmpeg.tar.xz \
    && cd ffmpeg-6.1.2 \
    && ./configure \
        --prefix=/usr/local \
        --enable-cuda --enable-cuvid --enable-nvdec --enable-nvenc \
        --enable-nonfree --enable-libnpp \
        --extra-cflags="-I/usr/local/cuda/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64" \
        --disable-doc --disable-debug --enable-gpl \
    && make -j$(nproc) && make install \
    && rm -rf /tmp/ffmpeg*

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

# Remove conda's non-GPU ffmpeg so our build is used
RUN rm -f /opt/conda/bin/ffmpeg /opt/conda/bin/ffprobe 2>/dev/null || true

ENV PYTHONPATH="/opt/wham:${PYTHONPATH}"

COPY . .

CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "-Q", "wham_3d", "-c", "1", "--prefetch-multiplier=1"]

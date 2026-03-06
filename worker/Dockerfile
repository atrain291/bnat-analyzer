FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu>=1.17.0

ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib:/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib:${LD_LIBRARY_PATH}"

COPY . .

RUN mkdir -p /app/uploads

CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "-Q", "video_pipeline", "-c", "1", "--prefetch-multiplier=1"]

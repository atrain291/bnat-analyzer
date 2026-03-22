import os
from celery import Celery

redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")

app = Celery(
    "bharatanatyam_analyzer",
    broker=redis_url,
    backend=redis_url,
    include=["app.tasks.video_pipeline", "app.tasks.detect_dancers", "app.tasks.multi_angle_pipeline"],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
)

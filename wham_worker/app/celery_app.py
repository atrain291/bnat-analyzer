import os
from celery import Celery

REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")

app = Celery(
    "wham_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)

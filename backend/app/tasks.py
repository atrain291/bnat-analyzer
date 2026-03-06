from celery import Celery
from app.config import settings

celery_app = Celery(
    "bharatanatyam_analyzer",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "worker.app.tasks.video_pipeline.run_pipeline": {"queue": "video_pipeline"},
        "worker.app.tasks.detect_dancers.run_detection": {"queue": "video_pipeline"},
    },
)


def dispatch_detection(performance_id: int, video_path: str) -> str:
    result = celery_app.send_task(
        "worker.app.tasks.detect_dancers.run_detection",
        args=[performance_id, video_path],
        queue="video_pipeline",
    )
    return result.id


def dispatch_pipeline(performance_id: int, video_path: str, selected_tracks: list[dict] | None = None) -> str:
    result = celery_app.send_task(
        "worker.app.tasks.video_pipeline.run_pipeline",
        args=[performance_id, video_path],
        kwargs={"selected_tracks": selected_tracks},
        queue="video_pipeline",
    )
    return result.id

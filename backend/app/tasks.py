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
        "worker.app.tasks.transcode.run_transcode": {"queue": "video_pipeline"},
        "sam2_worker.app.tasks.run_sam2_tracking": {"queue": "sam2_tracking"},
    },
)


def dispatch_detection(performance_id: int, video_path: str) -> str:
    result = celery_app.send_task(
        "worker.app.tasks.detect_dancers.run_detection",
        args=[performance_id, video_path],
        queue="video_pipeline",
    )
    return result.id


def dispatch_pipeline(performance_id: int, video_path: str, selected_tracks: list[dict] | None = None,
                      resume: bool = False) -> str:
    result = celery_app.send_task(
        "worker.app.tasks.video_pipeline.run_pipeline",
        args=[performance_id, video_path],
        kwargs={"selected_tracks": selected_tracks, "resume": resume},
        queue="video_pipeline",
    )
    return result.id


def dispatch_transcode(performance_id: int, video_path: str) -> str:
    result = celery_app.send_task(
        "worker.app.tasks.transcode.run_transcode",
        args=[performance_id, video_path],
        queue="video_pipeline",
    )
    return result.id


def dispatch_sam2(performance_id: int, video_path: str, start_timestamp_ms: int,
                  click_prompts: list, selected_tracks: list) -> str:
    result = celery_app.send_task(
        "sam2_worker.app.tasks.run_sam2_tracking",
        args=[performance_id, video_path, start_timestamp_ms, click_prompts, selected_tracks],
        queue="sam2_tracking",
    )
    return result.id

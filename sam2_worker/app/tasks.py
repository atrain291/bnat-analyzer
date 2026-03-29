"""SAM 2 tracking Celery task."""

import gc
import logging
import os
import traceback

from celery import Celery

from app.celery_app import app
from app.db import get_session
from app.models import Performance, PerformanceDancer, TrackingFrame

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
_celery_sender = Celery("sam2_dispatch", broker=REDIS_URL)

BATCH_SIZE = 500


def _make_cancel_checker(performance_id: int):
    """Return a callable that checks Redis for a cancellation flag."""
    import redis
    r = redis.from_url(REDIS_URL)

    def is_cancelled():
        return r.exists(f"cancel:{performance_id}") > 0

    return is_cancelled


@app.task(bind=True, name="sam2_worker.app.tasks.run_sam2_tracking")
def run_sam2_tracking(
    self,
    performance_id: int,
    video_path: str,
    start_timestamp_ms: int,
    click_prompts: list,
    selected_tracks: list,
):
    """Run SAM 2 video segmentation and store tracking frames.

    After tracking completes, dispatches the pose estimation pipeline
    to the main worker.
    """
    logger.info("SAM 2 tracking starting for performance %d", performance_id)

    try:
        # Update status to tracking
        with get_session() as session:
            perf = session.query(Performance).filter(
                Performance.id == performance_id,
            ).first()
            if perf:
                perf.status = "tracking"
                perf.pipeline_progress = {
                    "stage": "tracking",
                    "pct": 0.0,
                    "message": "Loading SAM 2 model...",
                }

        is_cancelled = _make_cancel_checker(performance_id)

        def progress_callback(current, total):
            pct = (current / max(total, 1)) * 100
            with get_session() as session:
                perf = session.query(Performance).filter(
                    Performance.id == performance_id,
                ).first()
                if perf:
                    perf.pipeline_progress = {
                        "stage": "tracking",
                        "pct": round(pct, 1),
                        "message": f"Tracking dancers... frame {current}/{total}",
                    }

        # Run SAM 2 inference
        from app.inference import run_sam2_tracking as _run_tracking

        results = _run_tracking(
            video_path,
            start_timestamp_ms,
            click_prompts,
            is_cancelled=is_cancelled,
            progress_callback=progress_callback,
        )

        # Store tracking frames in batches
        with get_session() as session:
            batch = []
            for r in results:
                batch.append(TrackingFrame(
                    performance_id=performance_id,
                    dancer_index=r["dancer_index"],
                    timestamp_ms=r["timestamp_ms"],
                    bbox=r["bbox"],
                    mask_iou=r["mask_iou"],
                ))
                if len(batch) >= BATCH_SIZE:
                    session.add_all(batch)
                    session.flush()
                    batch = []
            if batch:
                session.add_all(batch)

        logger.info(
            "Stored %d tracking frames for performance %d",
            len(results), performance_id,
        )

        # Dispatch pose pipeline to main worker
        _celery_sender.send_task(
            "worker.app.tasks.video_pipeline.run_pipeline",
            args=[performance_id, video_path],
            kwargs={"selected_tracks": selected_tracks},
            queue="video_pipeline",
        )
        logger.info("Dispatched pose pipeline for performance %d", performance_id)

        del results
        gc.collect()

        return {
            "performance_id": performance_id,
            "status": "tracked",
        }

    except Exception as e:
        logger.error(
            "SAM 2 tracking failed for performance %d: %s\n%s",
            performance_id, e, traceback.format_exc(),
        )
        with get_session() as session:
            perf = session.query(Performance).filter(
                Performance.id == performance_id,
            ).first()
            if perf:
                perf.status = "failed"
                perf.error = f"SAM 2 tracking failed: {str(e)[:1000]}"
        return {
            "performance_id": performance_id,
            "status": "failed",
            "error": str(e)[:500],
        }

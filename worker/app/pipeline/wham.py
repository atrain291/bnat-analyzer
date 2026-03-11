"""WHAM 3D dispatch — sends tasks to the separate wham_worker container.

The wham_worker runs PyTorch 1.13.1 + CUDA 11.6 with WHAM inference.
This module is a thin Celery client that dispatches fire-and-forget tasks
to the wham_3d queue. The main pipeline completes immediately with 2D
scores; WHAM enriches frames asynchronously in the background.
"""

import logging
import os

from celery import Celery

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")

# Lightweight Celery app just for sending tasks (no worker, no imports)
_celery_sender = Celery("wham_dispatch", broker=REDIS_URL)


def dispatch_wham_3d(performance_id: int, video_path: str, video_info: dict) -> bool:
    """Fire-and-forget dispatch of WHAM 3D estimation to the wham_worker.

    Args:
        performance_id: ID of the performance to process.
        video_path: Path to the video file (must be on shared volume).
        video_info: Dict with width, height, fps.

    Returns:
        True if task was dispatched, False on error.
    """
    try:
        _celery_sender.send_task(
            "wham_worker.app.tasks.run_wham_3d",
            args=[performance_id, video_path, video_info],
            queue="wham_3d",
        )
        logger.info("Dispatched WHAM 3D task for performance %d", performance_id)
        return True
    except Exception as e:
        logger.warning("Failed to dispatch WHAM 3D task: %s", e)
        return False

import logging
import subprocess
import traceback

from celery import current_task

from app.celery_app import app
from app.db import get_session
from app.models.performance import Performance, DetectedPerson
from app.pipeline.ingest import extract_metadata
from app.pipeline.pose import run_detection_pass
from app.pipeline.tracker import run_tracker

logger = logging.getLogger(__name__)

DETECTION_FRAMES = 50


def _update_progress(performance_id: int, stage: str, pct: float, **extra):
    progress = {"stage": stage, "pct": round(pct, 1), **extra}
    with get_session() as session:
        perf = session.query(Performance).filter(Performance.id == performance_id).first()
        if perf:
            perf.pipeline_progress = progress
            perf.status = "detecting"
    current_task.update_state(state="PROGRESS", meta=progress)


def _save_detection_frame(video_path: str, output_path: str):
    """Extract first frame as JPEG for the selection screen."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path,
    ]
    subprocess.run(cmd, check=True)


@app.task(bind=True, name="worker.app.tasks.detect_dancers.run_detection")
def run_detection(self, performance_id: int, video_path: str):
    logger.info(f"Starting detection for performance {performance_id}: {video_path}")

    try:
        _update_progress(performance_id, "ingest", 5.0)
        metadata = extract_metadata(video_path)

        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.duration_ms = metadata["duration_ms"]

        # Save detection frame
        frame_path = video_path.rsplit(".", 1)[0] + "_detection.jpg"
        _save_detection_frame(video_path, frame_path)
        video_key = video_path.split("/")[-1]
        detection_frame_key = video_key.rsplit(".", 1)[0] + "_detection.jpg"

        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.detection_frame_url = f"/uploads/{detection_frame_key}"

        # Run detection pass
        total_detect = min(DETECTION_FRAMES, metadata["total_frames"])

        def detection_progress(current: int, total: int):
            pct = 10.0 + (current / max(total, 1)) * 80.0
            _update_progress(performance_id, "detection", pct, frame=current, total_frames=total)

        _update_progress(performance_id, "detection", 10.0, frame=0, total_frames=total_detect)
        all_frames = run_detection_pass(video_path, metadata, max_frames=DETECTION_FRAMES, progress_callback=detection_progress)

        # Run tracker
        _update_progress(performance_id, "detection", 92.0)
        persons = run_tracker(all_frames, min_frame_ratio=0.2)

        # Store detected persons
        with get_session() as session:
            for person in persons:
                dp = DetectedPerson(
                    performance_id=performance_id,
                    track_id=person["track_id"],
                    bbox=person["bbox"],
                    representative_pose=person["representative_pose"],
                    frame_count=person["frame_count"],
                    area=person["area"],
                )
                session.add(dp)

        # Set status to awaiting_selection
        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "awaiting_selection"
                perf.pipeline_progress = {"stage": "awaiting_selection", "pct": 100.0}

        logger.info(f"Detection complete for performance {performance_id}: {len(persons)} persons found")
        return {"status": "awaiting_selection", "persons": len(persons)}

    except Exception as e:
        logger.error(f"Detection failed for performance {performance_id}: {e}\n{traceback.format_exc()}")
        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "failed"
                perf.error = str(e)[:2000]
        raise

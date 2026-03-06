import logging
import traceback

from celery import current_task

from app.celery_app import app
from app.db import get_session
from app.models.performance import Performance, Frame, Analysis
from app.pipeline.ingest import extract_metadata
from app.pipeline.pose import run_pose_estimation
from app.pipeline.angles import summarize_pose_statistics
from app.pipeline.llm import generate_coaching_feedback

logger = logging.getLogger(__name__)

BATCH_SIZE = 300


def _update_progress(performance_id: int, stage: str, pct: float, **extra):
    progress = {"stage": stage, "pct": round(pct, 1), **extra}

    with get_session() as session:
        perf = session.query(Performance).filter(Performance.id == performance_id).first()
        if perf:
            perf.pipeline_progress = progress
            perf.status = "processing"

    current_task.update_state(
        state="PROGRESS",
        meta=progress,
    )


@app.task(bind=True, name="worker.app.tasks.video_pipeline.run_pipeline")
def run_pipeline(self, performance_id: int, video_path: str):
    logger.info(f"Starting pipeline for performance {performance_id}: {video_path}")

    try:
        # Stage 1: Ingest - extract video metadata
        _update_progress(performance_id, "ingest", 5.0)
        metadata = extract_metadata(video_path)

        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.duration_ms = metadata["duration_ms"]

        # Stage 2: Pose estimation
        total_frames = metadata["total_frames"]

        def pose_progress(current_frame: int, total: int):
            pct = 20.0 + (current_frame / max(total, 1)) * 63.0
            _update_progress(
                performance_id,
                "pose_estimation",
                pct,
                frame=current_frame,
                total_frames=total,
            )

        _update_progress(performance_id, "pose_estimation", 20.0, frame=0, total_frames=total_frames)
        frames_data = run_pose_estimation(video_path, metadata, progress_callback=pose_progress)

        # Store frames in batches
        with get_session() as session:
            batch = []
            for fd in frames_data:
                batch.append(Frame(
                    performance_id=performance_id,
                    timestamp_ms=fd["timestamp_ms"],
                    dancer_pose=fd["dancer_pose"],
                ))
                if len(batch) >= BATCH_SIZE:
                    session.add_all(batch)
                    session.flush()
                    batch = []
            if batch:
                session.add_all(batch)
                session.flush()

        # Stage 3: Compute pose statistics
        _update_progress(performance_id, "pose_analysis", 83.0)
        pose_summary = summarize_pose_statistics(frames_data)

        # Stage 4: LLM coaching synthesis
        _update_progress(performance_id, "llm_synthesis", 85.0)

        # Get performance metadata for LLM context
        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            item_name = perf.item_name if perf else None
            item_type = perf.item_type if perf else None
            talam = perf.talam if perf else None

        coaching_text = generate_coaching_feedback(
            frame_count=len(frames_data),
            duration_ms=metadata["duration_ms"],
            item_name=item_name,
            item_type=item_type,
            talam=talam,
            pose_summary=pose_summary,
        )

        with get_session() as session:
            analysis = Analysis(
                performance_id=performance_id,
                llm_summary=coaching_text,
            )
            session.add(analysis)

        # Stage 4: Complete
        _update_progress(performance_id, "complete", 100.0)

        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "complete"

        logger.info(f"Pipeline complete for performance {performance_id}: {len(frames_data)} frames")
        return {"status": "complete", "frames": len(frames_data)}

    except Exception as e:
        logger.error(f"Pipeline failed for performance {performance_id}: {e}\n{traceback.format_exc()}")

        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "failed"
                perf.error = str(e)[:2000]

        raise

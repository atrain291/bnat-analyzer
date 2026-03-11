"""WHAM Celery task — dispatched by the main worker pipeline.

Reads 2D pose data from Postgres, runs WHAM 3D reconstruction,
writes joints_3d/foot_contact/world_position back to frames table.
"""
import logging

from app.celery_app import app
from app.db import get_session
from app.models import Frame, Performance

logger = logging.getLogger(__name__)


@app.task(name="wham_worker.app.tasks.run_wham_3d", bind=True)
def run_wham_3d(self, performance_id: int, video_path: str, video_info: dict) -> dict:
    """Run WHAM 3D estimation for a performance.

    Reads RTMPose 2D poses from frames table, runs WHAM per dancer,
    writes 3D data back to the same frame rows.
    """
    from app.inference import run_wham_inference, release_model

    logger.info("WHAM 3D starting for performance %d", performance_id)

    try:
        with get_session() as session:
            frames = (
                session.query(Frame)
                .filter(Frame.performance_id == performance_id)
                .order_by(Frame.performance_dancer_id, Frame.timestamp_ms)
                .all()
            )

            if not frames:
                logger.warning("No frames found for performance %d", performance_id)
                return {"performance_id": performance_id, "status": "skipped", "reason": "no_frames"}

            # Group frames by dancer
            dancers = {}
            for f in frames:
                key = f.performance_dancer_id  # None for single-dancer
                if key not in dancers:
                    dancers[key] = []
                dancers[key].append(f)

            total_updated = 0

            for dancer_id, dancer_frames in dancers.items():
                label = f"dancer {dancer_id}" if dancer_id else "solo"
                poses = [f.dancer_pose if f.dancer_pose else {} for f in dancer_frames]
                frame_db_ids = [f.id for f in dancer_frames]

                valid_count = sum(1 for p in poses if p)
                if valid_count < 10:
                    logger.info("Skipping %s: only %d valid poses", label, valid_count)
                    continue

                logger.info("Running WHAM for %s (%d valid poses)...", label, valid_count)
                wham_result = run_wham_inference(video_path, poses, video_info)

                if wham_result is None:
                    logger.warning("WHAM returned None for %s", label)
                    continue

                # Map WHAM output frames back to DB frame IDs
                wham_frame_indices = wham_result.get("frame_ids", list(range(wham_result["frame_count"])))
                updated = 0

                for t in range(wham_result["frame_count"]):
                    if t >= len(wham_frame_indices):
                        break
                    src_idx = wham_frame_indices[t]
                    if src_idx >= len(frame_db_ids):
                        break

                    db_frame_id = frame_db_ids[src_idx]

                    # Build update dict
                    update = {}
                    j3d = wham_result["joints_3d"][t] if t < len(wham_result["joints_3d"]) else None
                    if j3d is not None:
                        update["joints_3d"] = j3d

                    wp = wham_result["world_positions"][t] if t < len(wham_result["world_positions"]) else None
                    if wp is not None:
                        update["world_position"] = wp

                    fc = wham_result["foot_contacts"][t] if t < len(wham_result["foot_contacts"]) else None
                    if fc is not None:
                        update["foot_contact"] = fc

                    if update:
                        session.query(Frame).filter(Frame.id == db_frame_id).update(update)
                        updated += 1

                    if updated % 500 == 0 and updated > 0:
                        session.flush()

                total_updated += updated
                logger.info("Updated %d frames with 3D data for %s", updated, label)

        # Release GPU
        release_model()

        logger.info("WHAM 3D complete for performance %d: %d frames updated", performance_id, total_updated)
        return {"performance_id": performance_id, "status": "complete", "frames_updated": total_updated}

    except Exception as e:
        logger.exception("WHAM 3D failed for performance %d", performance_id)
        release_model()
        return {"performance_id": performance_id, "status": "failed", "error": str(e)[:500]}

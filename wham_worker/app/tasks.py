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
            # Only load columns we need (avoids fetching heavy JSON we won't use)
            rows = (
                session.query(
                    Frame.id, Frame.performance_dancer_id,
                    Frame.timestamp_ms, Frame.dancer_pose
                )
                .filter(Frame.performance_id == performance_id)
                .order_by(Frame.performance_dancer_id, Frame.timestamp_ms)
                .all()
            )

            if not rows:
                logger.warning("No frames found for performance %d", performance_id)
                return {"performance_id": performance_id, "status": "skipped", "reason": "no_frames"}

            # Group by dancer — each row is (id, pd_id, ts, pose)
            dancers = {}
            for row in rows:
                key = row[1]  # performance_dancer_id
                if key not in dancers:
                    dancers[key] = []
                dancers[key].append(row)

            total_updated = 0

            for dancer_id, dancer_rows in dancers.items():
                label = f"dancer {dancer_id}" if dancer_id else "solo"
                poses = [r[3] if r[3] else {} for r in dancer_rows]  # dancer_pose
                frame_db_ids = [r[0] for r in dancer_rows]  # frame id

                valid_count = sum(1 for p in poses if p)
                if valid_count < 10:
                    logger.info("Skipping %s: only %d valid poses", label, valid_count)
                    continue

                logger.info("Running WHAM for %s (%d valid poses)...", label, valid_count)
                wham_result = run_wham_inference(video_path, poses, video_info)

                if wham_result is None:
                    logger.warning("WHAM returned None for %s", label)
                    continue

                # Map WHAM output frames back to DB frame IDs — batch update
                wham_frame_indices = wham_result.get("frame_ids", list(range(wham_result["frame_count"])))
                bulk_updates = []

                for t in range(wham_result["frame_count"]):
                    if t >= len(wham_frame_indices):
                        break
                    src_idx = wham_frame_indices[t]
                    if src_idx >= len(frame_db_ids):
                        break

                    row = {"id": frame_db_ids[src_idx]}
                    j3d = wham_result["joints_3d"][t] if t < len(wham_result["joints_3d"]) else None
                    if j3d is not None:
                        row["joints_3d"] = j3d
                    wp = wham_result["world_positions"][t] if t < len(wham_result["world_positions"]) else None
                    if wp is not None:
                        row["world_position"] = wp
                    fc = wham_result["foot_contacts"][t] if t < len(wham_result["foot_contacts"]) else None
                    if fc is not None:
                        row["foot_contact"] = fc

                    if len(row) > 1:  # has data beyond just 'id'
                        bulk_updates.append(row)

                # Bulk update in batches of 1000
                updated = 0
                BATCH = 1000
                for i in range(0, len(bulk_updates), BATCH):
                    batch = bulk_updates[i:i + BATCH]
                    session.bulk_update_mappings(Frame, batch)
                    session.flush()
                    updated += len(batch)

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

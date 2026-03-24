"""WHAM Celery task — dispatched by the main worker pipeline.

Reads 2D pose data from Postgres, runs WHAM 3D reconstruction,
writes joints_3d/foot_contact/world_position back to frames table.
"""
import logging
import time

from sqlalchemy.exc import OperationalError

from app.celery_app import app
from app.db import get_session
from app.models import Frame

logger = logging.getLogger(__name__)

BATCH = 1000
DEADLOCK_RETRIES = 3
DEADLOCK_PGCODE = "40P01"


def _write_dancer_updates(bulk_updates: list[dict], label: str) -> int:
    """Write 3D data for one dancer with deadlock retry.

    Opens a fresh session, commits, and closes. Retries up to
    DEADLOCK_RETRIES times on Postgres deadlock (pgcode 40P01).
    """
    # Sort by frame id to ensure consistent lock ordering across transactions
    bulk_updates.sort(key=lambda r: r["id"])

    for attempt in range(1, DEADLOCK_RETRIES + 1):
        try:
            with get_session() as session:
                for i in range(0, len(bulk_updates), BATCH):
                    batch = bulk_updates[i:i + BATCH]
                    session.bulk_update_mappings(Frame, batch)
                    session.flush()
            return len(bulk_updates)
        except OperationalError as e:
            pgcode = getattr(e.orig, "pgcode", None) if hasattr(e, "orig") else None
            if pgcode == DEADLOCK_PGCODE and attempt < DEADLOCK_RETRIES:
                wait = 2 ** (attempt - 1)
                logger.warning("Deadlock writing 3D data for %s (attempt %d/%d), retrying in %ds",
                               label, attempt, DEADLOCK_RETRIES, wait)
                time.sleep(wait)
            else:
                raise

    return 0


@app.task(name="wham_worker.app.tasks.run_wham_3d", bind=True)
def run_wham_3d(self, performance_id: int, video_path: str, video_info: dict) -> dict:
    """Run WHAM 3D estimation for a performance.

    Reads RTMPose 2D poses from frames table, runs WHAM per dancer,
    writes 3D data back to the same frame rows.
    """
    from app.inference import run_wham_inference, release_model

    logger.info("WHAM 3D starting for performance %d", performance_id)

    total_updated = 0

    try:
        # --- Phase 1: Read frames (short-lived session, no locks held during inference) ---
        with get_session() as session:
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

        # --- Phase 2: Inference (no DB connection) ---
        dancer_results = {}
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

            # Map WHAM output frames back to DB frame IDs
            wham_frame_indices = wham_result.get("frame_ids", list(range(wham_result["frame_count"])))
            bulk_updates = []

            for t in range(wham_result["frame_count"]):
                if t >= len(wham_frame_indices):
                    break
                src_idx = wham_frame_indices[t]
                if src_idx >= len(frame_db_ids):
                    logger.warning("WHAM frame_id %d out of range (max %d), skipping", src_idx, len(frame_db_ids))
                    continue

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

            if bulk_updates:
                dancer_results[dancer_id] = (label, bulk_updates)

        # --- Phase 3: Write back (fresh session per dancer, with deadlock retry) ---
        for dancer_id, (label, bulk_updates) in dancer_results.items():
            updated = _write_dancer_updates(bulk_updates, label)
            total_updated += updated
            logger.info("Updated %d frames with 3D data for %s", updated, label)

        # Release GPU
        release_model()

        logger.info("WHAM 3D complete for performance %d: %d frames updated", performance_id, total_updated)
        return {"performance_id": performance_id, "status": "complete", "frames_updated": total_updated}

    except Exception as e:
        logger.exception("WHAM 3D failed for performance %d", performance_id)
        release_model()
        return {"performance_id": performance_id, "status": "failed", "error": str(e)[:500],
                "frames_updated": total_updated}

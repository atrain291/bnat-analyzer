"""WHAM Celery task — dispatched by the main worker pipeline.

Reads 2D pose data from Postgres, runs WHAM 3D reconstruction,
writes joints_3d/foot_contact/world_position back to frames table.
"""
import gc
import logging
import time

from sqlalchemy.exc import OperationalError

from app.celery_app import app
from app.db import get_session
from app.models import Frame

logger = logging.getLogger(__name__)

MIN_BIO_RATIOS = 3
BATCH = 1000
DEADLOCK_RETRIES = 3
DEADLOCK_PGCODE = "40P01"


def _compute_3d_biometrics(joints_3d: list) -> dict | None:
    """Compute body-proportion ratios from SMPL 24-joint 3D positions.

    Returns dict of ratios or None if too few computable.
    SMPL: 0=pelvis, 1=L_hip, 2=R_hip, 4=L_knee, 5=R_knee,
    7=L_ankle, 8=R_ankle, 12=neck, 15=head,
    16=L_shoulder, 17=R_shoulder, 18=L_elbow, 19=R_elbow, 20=L_wrist, 21=R_wrist.
    """
    if not joints_3d or len(joints_3d) < 22:
        return None

    def dist(i, j):
        a, b = joints_3d[i], joints_3d[j]
        if not a or not b or len(a) < 3 or len(b) < 3:
            return None
        d = ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5
        return d if d > 1e-6 else None

    def avg(a, b):
        if a and b: return (a + b) / 2
        return a or b

    def ratio(n, d):
        if n is None or d is None or d < 1e-6: return None
        return n / d

    shoulder_w = dist(16, 17)
    hip_w = dist(1, 2)
    # Torso: shoulder midpoint to hip midpoint
    s16, s17, h1, h2 = joints_3d[16], joints_3d[17], joints_3d[1], joints_3d[2]
    torso_len = None
    if s16 and s17 and h1 and h2 and all(len(j) >= 3 for j in (s16, s17, h1, h2)):
        sm = [(s16[k]+s17[k])/2 for k in range(3)]
        hm = [(h1[k]+h2[k])/2 for k in range(3)]
        d = sum((sm[k]-hm[k])**2 for k in range(3)) ** 0.5
        torso_len = d if d > 1e-6 else None

    avg_thigh = avg(dist(1, 4), dist(2, 5))
    avg_shin = avg(dist(4, 7), dist(5, 8))
    avg_leg = (avg_thigh + avg_shin) if (avg_thigh and avg_shin) else None
    avg_upper_arm = avg(dist(16, 18), dist(17, 19))
    avg_lower_arm = avg(dist(18, 20), dist(19, 21))
    l_arm = (dist(16, 18) or 0) + (dist(18, 20) or 0) if dist(16, 18) and dist(18, 20) else None
    r_arm = (dist(17, 19) or 0) + (dist(19, 21) or 0) if dist(17, 19) and dist(19, 21) else None
    avg_arm_len = avg(l_arm, r_arm)
    head_neck = dist(15, 12)

    ratios = {}
    for name, val in [
        ("shoulder_hip_ratio", ratio(shoulder_w, hip_w)),
        ("torso_leg_ratio", ratio(torso_len, avg_leg)),
        ("upper_lower_arm_ratio", ratio(avg_upper_arm, avg_lower_arm)),
        ("thigh_shin_ratio", ratio(avg_thigh, avg_shin)),
        ("head_shoulder_ratio", ratio(head_neck, shoulder_w)),
        ("arm_body_ratio", ratio(avg_arm_len, torso_len)),
    ]:
        if val is not None:
            ratios[name] = round(val, 4)

    return ratios if len(ratios) >= MIN_BIO_RATIOS else None


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

            # Compute 3D biometric ratios from sampled frames
            bio_samples = []
            for t in range(0, wham_result["frame_count"], 5):
                j3d = wham_result["joints_3d"][t] if t < len(wham_result["joints_3d"]) else None
                if j3d:
                    bio = _compute_3d_biometrics(j3d)
                    if bio:
                        bio_samples.append(bio)
            if bio_samples:
                avg_bio = {}
                for key in bio_samples[0]:
                    vals = [s[key] for s in bio_samples if key in s]
                    if vals:
                        avg_bio[key] = round(sum(vals) / len(vals), 4)
                logger.info("3D biometrics for %s (%d samples): %s", label, len(bio_samples), avg_bio)

        # --- Phase 3: Write back (fresh session per dancer, with deadlock retry) ---
        for dancer_id, (label, bulk_updates) in dancer_results.items():
            updated = _write_dancer_updates(bulk_updates, label)
            total_updated += updated
            logger.info("Updated %d frames with 3D data for %s", updated, label)

        # Release GPU and free large data structures
        del dancer_results, dancers, rows
        release_model()
        gc.collect()

        logger.info("WHAM 3D complete for performance %d: %d frames updated", performance_id, total_updated)
        return {"performance_id": performance_id, "status": "complete", "frames_updated": total_updated}

    except Exception as e:
        logger.exception("WHAM 3D failed for performance %d", performance_id)
        release_model()
        gc.collect()
        return {"performance_id": performance_id, "status": "failed", "error": str(e)[:500],
                "frames_updated": total_updated}

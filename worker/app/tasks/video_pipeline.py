import logging
import time
import traceback

from celery import current_task

from app.celery_app import app
from app.db import get_session
from app.models.performance import Performance, PerformanceDancer, DetectedPerson, Frame, Analysis, JointAngleState, BalanceMetrics
from app.pipeline.ingest import extract_metadata
from app.pipeline.pose import run_pose_estimation, run_pose_estimation_multi
from app.pipeline.pose_config import POSE_FRAME_SKIP
from app.pipeline.angles import compute_frame_angles, OnlineAngleAccumulator
from app.pipeline.llm import generate_coaching_feedback
from app.pipeline.scoring import compute_scores
from app.pipeline.beat_detection import run_beat_analysis, detect_foot_strikes_from_series, score_rhythm_sync
from app.pipeline.wham import dispatch_wham_3d

logger = logging.getLogger(__name__)

BATCH_SIZE = 300
PROGRESS_DB_INTERVAL = 2.0  # seconds between Postgres progress writes


def _make_cancel_checker(performance_id: int):
    """Return a callable that checks Redis for a cancellation flag."""
    import redis
    import os
    r = redis.from_url(os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"))

    def is_cancelled() -> bool:
        return r.exists(f"cancel:{performance_id}") > 0

    return is_cancelled


def _make_progress_updater(performance_id: int, status: str = "processing"):
    """Return a throttled progress-update callable.

    The returned function always pushes to Redis (cheap) but only writes
    to Postgres when the stage changes or PROGRESS_DB_INTERVAL has elapsed.
    """
    state = {"last_stage": None, "last_message": None, "last_db_time": 0.0}

    def update(stage: str, pct: float, final_status: str | None = None, **extra):
        progress = {"stage": stage, "pct": round(pct, 1), **extra}

        now = time.monotonic()
        stage_changed = stage != state["last_stage"]
        msg = extra.get("message")
        message_changed = msg and msg != state["last_message"] and "frame" not in extra
        elapsed = now - state["last_db_time"]

        if stage_changed or message_changed or elapsed >= PROGRESS_DB_INTERVAL or final_status:
            with get_session() as session:
                perf = session.query(Performance).filter(
                    Performance.id == performance_id
                ).first()
                if perf:
                    perf.pipeline_progress = progress
                    perf.status = final_status or status
            state["last_stage"] = stage
            state["last_message"] = extra.get("message")
            state["last_db_time"] = now

        current_task.update_state(state="PROGRESS", meta=progress)

    return update


def _store_frames_and_metrics(session, performance_id, performance_dancer_id, frames_iter):
    """Store frame rows with angles and balance metrics in a single streaming pass.

    Accepts any iterable of frame dicts (list or generator).  Computes angles
    once per frame and accumulates running statistics via OnlineAngleAccumulator.

    Returns (frame_count, accumulator) so the caller can get pose_summary
    and foot-flatness data without a second pass.
    """
    accumulator = OnlineAngleAccumulator()
    # pending_angles holds (frame_obj, angles_dict, fd) for frames awaiting ID assignment
    pending_angles: list[tuple] = []
    frame_batch = []
    frame_count = 0

    def _flush_batch():
        """Flush accumulated frame rows, then create angle/balance rows for them."""
        nonlocal frame_batch, pending_angles
        if frame_batch:
            session.add_all(frame_batch)
            session.flush()  # assigns frame IDs
            frame_batch = []

        if not pending_angles:
            return
        angle_objs = []
        balance_objs = []
        for frame_obj, angles, fd in pending_angles:
            pose = fd.get("dancer_pose", {})
            angle_objs.append(JointAngleState(
                frame_id=frame_obj.id,
                aramandi_angle=angles.get("avg_knee_angle"),
                torso_uprightness=angles.get("torso_angle"),
                arm_extension_left=angles.get("arm_extension_left"),
                arm_extension_right=angles.get("arm_extension_right"),
                hip_symmetry=angles.get("hip_symmetry"),
                knee_angle_3d=angles.get("knee_angle_3d"),
                torso_angle_3d=angles.get("torso_angle_3d"),
                hip_abduction_left=angles.get("hip_abduction_left"),
                hip_abduction_right=angles.get("hip_abduction_right"),
                torso_twist=angles.get("torso_twist"),
                all_angles=angles,
            ))

            lh = pose.get("left_hip", {})
            rh = pose.get("right_hip", {})
            com_x = None
            com_y = None
            if lh.get("confidence", 0) > 0.3 and rh.get("confidence", 0) > 0.3:
                com_x = (lh["x"] + rh["x"]) / 2
                com_y = (lh["y"] + rh["y"]) / 2

            hip_sym = angles.get("hip_symmetry")
            weight_dist = None
            if hip_sym is not None:
                if lh.get("confidence", 0) > 0.3 and rh.get("confidence", 0) > 0.3:
                    sign = 1.0 if lh["y"] > rh["y"] else -1.0
                    weight_dist = max(-1.0, min(1.0, sign * hip_sym * 5.0))

            stability_components = []
            torso = angles.get("torso_angle")
            if torso is not None:
                stability_components.append(max(0.0, min(1.0, 1.0 - (torso - 2) / 13)))
            if hip_sym is not None:
                stability_components.append(max(0.0, min(1.0, 1.0 - (hip_sym - 0.02) / 0.13)))
            stability = None
            if stability_components:
                stability = sum(stability_components) / len(stability_components)

            j3d = fd.get("joints_3d")
            com_3d_x = com_3d_y = com_3d_z = None
            if j3d and len(j3d) >= 1:
                com_3d_x = float(j3d[0][0])
                com_3d_y = float(j3d[0][1])
                com_3d_z = float(j3d[0][2])

            balance_objs.append(BalanceMetrics(
                frame_id=frame_obj.id,
                center_of_mass_x=com_x,
                center_of_mass_y=com_y,
                weight_distribution=weight_dist,
                stability_score=stability,
                center_of_mass_3d_x=com_3d_x,
                center_of_mass_3d_y=com_3d_y,
                center_of_mass_3d_z=com_3d_z,
            ))

        session.add_all(angle_objs)
        session.add_all(balance_objs)
        session.flush()
        pending_angles = []

    for fd in frames_iter:
        pose = fd.get("dancer_pose", {})
        timestamp_ms = fd["timestamp_ms"]

        frame_obj = Frame(
            performance_id=performance_id,
            performance_dancer_id=performance_dancer_id,
            timestamp_ms=timestamp_ms,
            dancer_pose=pose,
            left_hand=fd.get("left_hand"),
            right_hand=fd.get("right_hand"),
            face=fd.get("face"),
            joints_3d=fd.get("joints_3d"),
            world_position=fd.get("world_position"),
            foot_contact=fd.get("foot_contact"),
        )
        frame_batch.append(frame_obj)
        frame_count += 1

        if pose:
            angles = compute_frame_angles(
                pose,
                face=fd.get("face"),
                left_hand=fd.get("left_hand"),
                right_hand=fd.get("right_hand"),
                joints_3d=fd.get("joints_3d"),
            )
            if angles:
                accumulator.add_frame(angles, timestamp_ms=timestamp_ms, pose=pose)
                pending_angles.append((frame_obj, angles, fd))

        if len(frame_batch) >= BATCH_SIZE:
            _flush_batch()

    # Flush remaining
    _flush_batch()

    return frame_count, accumulator


def _analyze_dancer(performance_id, performance_dancer_id, label, frame_count,
                    pose_summary, metadata, beat_data=None, accumulator=None):
    """Run LLM coaching, scoring, and rhythm analysis for one dancer.

    Args:
        frame_count: Number of frames processed for this dancer.
        pose_summary: Pre-computed aggregate stats dict (from OnlineAngleAccumulator.summarize).
        accumulator: OnlineAngleAccumulator with foot-flatness data for rhythm scoring.
    """
    # Rhythm scoring: correlate foot strikes with audio onsets
    rhythm_score = None
    rhythm_details = {}
    if beat_data and beat_data.get("onset_timestamps_ms") and accumulator:
        foot_strikes = detect_foot_strikes_from_series(
            accumulator.foot_flatness_timestamps,
            accumulator.foot_flatness_values,
        )
        sync_result = score_rhythm_sync(beat_data["onset_timestamps_ms"], foot_strikes)
        rhythm_score = sync_result.get("rhythm_score")
        rhythm_details = sync_result
        if rhythm_score is not None:
            pose_summary["rhythm_score"] = rhythm_score
            pose_summary["rhythm_match_rate"] = sync_result.get("match_rate", 0)
            pose_summary["rhythm_avg_offset_ms"] = sync_result.get("avg_offset_ms")
            pose_summary["tempo_bpm"] = beat_data.get("tempo_bpm")

    with get_session() as session:
        perf = session.query(Performance).filter(Performance.id == performance_id).first()
        item_name = perf.item_name if perf else None
        item_type = perf.item_type if perf else None
        talam = perf.talam if perf else None

    coaching_text = generate_coaching_feedback(
        frame_count=frame_count,
        duration_ms=metadata["duration_ms"],
        item_name=item_name,
        item_type=item_type,
        talam=talam,
        pose_summary=pose_summary,
        dancer_label=label,
    )

    scores = compute_scores(pose_summary)

    # Include rhythm details in technique_scores
    if rhythm_details:
        scores["technique_scores"]["rhythm_details"] = rhythm_details
    if rhythm_score is not None:
        scores["technique_scores"]["inputs"]["rhythm_score"] = rhythm_score
        scores["technique_scores"]["inputs"]["tempo_bpm"] = beat_data.get("tempo_bpm")

    with get_session() as session:
        analysis = Analysis(
            performance_id=performance_id,
            performance_dancer_id=performance_dancer_id,
            llm_summary=coaching_text,
            aramandi_score=scores["aramandi_score"],
            upper_body_score=scores["upper_body_score"],
            symmetry_score=scores["symmetry_score"],
            rhythm_consistency_score=rhythm_score,
            overall_score=scores["overall_score"],
            technique_scores=scores["technique_scores"],
        )
        session.add(analysis)


@app.task(bind=True, name="worker.app.tasks.video_pipeline.run_pipeline")
def run_pipeline(self, performance_id: int, video_path: str, selected_tracks: list[dict] | None = None):
    logger.info(f"Starting pipeline for performance {performance_id}: {video_path}")
    update_progress = _make_progress_updater(performance_id, status="processing")

    try:
        # Stage 1: Ingest (metadata only — transcode already done in detection phase)
        update_progress("ingest", 7.0, message="Video ready")
        metadata = extract_metadata(video_path)

        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.duration_ms = metadata["duration_ms"]

        total_frames = metadata["total_frames"]
        is_cancelled = _make_cancel_checker(performance_id)

        # Stage 1b: Audio beat detection (fast, CPU-only, runs before pose estimation)
        update_progress("beat_detection", 10.0, message="Analyzing audio for beat onsets...")
        beat_data = None
        try:
            beat_data = run_beat_analysis(video_path, metadata)
            if beat_data:
                with get_session() as session:
                    perf = session.query(Performance).filter(Performance.id == performance_id).first()
                    if perf:
                        perf.beat_timestamps = beat_data["onset_timestamps_ms"]
                        perf.tempo_bpm = beat_data["tempo_bpm"]
                logger.info(f"Beat detection: {beat_data['onset_count']} onsets at {beat_data['tempo_bpm']} BPM")
        except Exception as e:
            logger.warning(f"Beat detection failed (non-fatal): {e}")

        if selected_tracks:
            # Multi-dancer mode — stream frames through per-dancer accumulators
            track_id_to_info = {t["track_id"]: t for t in selected_tracks}
            selected_ids = set(track_id_to_info.keys())

            # Fetch detection bboxes and appearance to seed the tracker
            seed_bboxes = {}
            seed_histograms = {}
            with get_session() as session:
                detected = session.query(DetectedPerson).filter(
                    DetectedPerson.performance_id == performance_id
                ).all()
                for dp in detected:
                    bbox = dp.bbox
                    seed_bboxes[dp.track_id] = (bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"])
                    if dp.color_histogram:
                        seed_histograms[dp.track_id] = dp.color_histogram

            def pose_progress(current_frame: int, total: int):
                pct = 20.0 + (current_frame / max(total, 1)) * 60.0
                update_progress("pose_estimation", pct, frame=current_frame, total_frames=total,
                                message=f"Estimating poses... frame {current_frame}/{total}")

            effective_total = total_frames // POSE_FRAME_SKIP
            update_progress("pose_estimation", 20.0, frame=0, total_frames=effective_total,
                            message="Loading pose model...")

            # Stream (track_id, frame_dict) tuples from the generator, routing
            # each frame to its dancer's per-dancer list for batch DB storage.
            per_dancer_frames: dict[int, list[dict]] = {tid: [] for tid in selected_ids}
            frame_gen = run_pose_estimation_multi(
                video_path, metadata, selected_ids, progress_callback=pose_progress,
                is_cancelled=is_cancelled, seed_bboxes=seed_bboxes,
                seed_histograms=seed_histograms if seed_histograms else None,
            )
            for tid, fd in frame_gen:
                per_dancer_frames[tid].append(fd)

            # Store frames + compute stats per dancer (single pass each, no recomputation)
            update_progress("pose_analysis", 80.0, message="Computing joint angles and storing frames...")
            dancer_results: dict[int, tuple[int, dict, OnlineAngleAccumulator]] = {}
            for track_id, frames_data in per_dancer_frames.items():
                info = track_id_to_info[track_id]
                pd_id = info["performance_dancer_id"]
                with get_session() as session:
                    frame_count, accumulator = _store_frames_and_metrics(
                        session, performance_id, pd_id, frames_data,
                    )
                pose_summary = accumulator.summarize()
                dancer_results[track_id] = (frame_count, pose_summary, accumulator)
                del frames_data  # allow GC before next dancer

            # Analyze each dancer with pre-computed stats (no second pass)
            num_dancers = len(selected_tracks)
            for i, track_id in enumerate(dancer_results):
                info = track_id_to_info[track_id]
                frame_count, pose_summary, accumulator = dancer_results[track_id]
                pct = 83.0 + (i / max(num_dancers, 1)) * 12.0
                dancer_label = info.get("label") or f"Dancer {i + 1}"
                update_progress("llm_synthesis", pct,
                                message=f"Generating coaching for {dancer_label}... ({i + 1}/{num_dancers})")
                if frame_count:
                    _analyze_dancer(
                        performance_id, info["performance_dancer_id"],
                        info.get("label"), frame_count, pose_summary, metadata,
                        beat_data=beat_data, accumulator=accumulator,
                    )

        else:
            # Single-dancer mode — stream directly from generator through storage
            def pose_progress(current_frame: int, total: int):
                pct = 20.0 + (current_frame / max(total, 1)) * 63.0
                update_progress("pose_estimation", pct, frame=current_frame, total_frames=total,
                                message=f"Estimating poses... frame {current_frame}/{total}")

            effective_total = total_frames // POSE_FRAME_SKIP
            update_progress("pose_estimation", 20.0, frame=0, total_frames=effective_total,
                            message="Loading pose model...")
            frame_gen = run_pose_estimation(video_path, metadata, progress_callback=pose_progress, is_cancelled=is_cancelled)

            with get_session() as session:
                frame_count, accumulator = _store_frames_and_metrics(session, performance_id, None, frame_gen)
            pose_summary = accumulator.summarize()

            update_progress("pose_analysis", 83.0, message="Computing joint angles...")
            update_progress("llm_synthesis", 85.0, message="Generating coaching feedback...")
            _analyze_dancer(
                performance_id, None, None, frame_count, pose_summary, metadata,
                beat_data=beat_data, accumulator=accumulator,
            )

        # Dispatch WHAM 3D estimation (fire-and-forget, runs in separate container)
        video_info = {"width": metadata.get("width", 1920), "height": metadata.get("height", 1080), "fps": metadata.get("fps", 30)}
        dispatch_wham_3d(performance_id, video_path, video_info)

        # Complete
        update_progress("scoring", 95.0, message="Computing final scores...")
        update_progress("complete", 100.0, final_status="complete", message="Analysis complete!")

        logger.info(f"Pipeline complete for performance {performance_id}")
        return {"status": "complete"}

    except Exception as e:
        logger.error(f"Pipeline failed for performance {performance_id}: {e}\n{traceback.format_exc()}")

        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "failed"
                perf.error = str(e)[:2000]

        raise

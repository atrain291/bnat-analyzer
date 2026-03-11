import logging
import traceback

from celery import current_task

from app.celery_app import app
from app.db import get_session
from app.models.performance import Performance, PerformanceDancer, DetectedPerson, Frame, Analysis, JointAngleState, BalanceMetrics
from app.pipeline.ingest import extract_metadata
from app.pipeline.pose import run_pose_estimation, run_pose_estimation_multi
from app.pipeline.angles import summarize_pose_statistics, compute_frame_angles
from app.pipeline.llm import generate_coaching_feedback
from app.pipeline.scoring import compute_scores
from app.pipeline.beat_detection import run_beat_analysis, detect_foot_strikes, score_rhythm_sync
from app.pipeline.wham import dispatch_wham_3d

logger = logging.getLogger(__name__)

BATCH_SIZE = 300


def _make_cancel_checker(performance_id: int):
    """Return a callable that checks Redis for a cancellation flag."""
    import redis
    import os
    r = redis.from_url(os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"))

    def is_cancelled() -> bool:
        return r.exists(f"cancel:{performance_id}") > 0

    return is_cancelled


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


def _store_frames_and_metrics(session, performance_id, performance_dancer_id, frames_data):
    """Store frame rows with angles and balance metrics for one dancer."""
    all_frames = []
    batch = []
    for fd in frames_data:
        frame_obj = Frame(
            performance_id=performance_id,
            performance_dancer_id=performance_dancer_id,
            timestamp_ms=fd["timestamp_ms"],
            dancer_pose=fd["dancer_pose"],
            left_hand=fd.get("left_hand"),
            right_hand=fd.get("right_hand"),
            face=fd.get("face"),
            joints_3d=fd.get("joints_3d"),
            world_position=fd.get("world_position"),
            foot_contact=fd.get("foot_contact"),
        )
        batch.append(frame_obj)
        all_frames.append((frame_obj, fd))
        if len(batch) >= BATCH_SIZE:
            session.add_all(batch)
            session.flush()
            batch = []
    if batch:
        session.add_all(batch)
        session.flush()

    angle_batch = []
    balance_batch = []
    for frame_obj, fd in all_frames:
        pose = fd.get("dancer_pose", {})
        if not pose:
            continue

        angles = compute_frame_angles(
            pose,
            face=fd.get("face"),
            left_hand=fd.get("left_hand"),
            right_hand=fd.get("right_hand"),
            joints_3d=fd.get("joints_3d"),
        )
        if not angles:
            continue

        angle_batch.append(JointAngleState(
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

        # 3D center of mass from WHAM joints (pelvis = joint 0)
        j3d = fd.get("joints_3d")
        com_3d_x = com_3d_y = com_3d_z = None
        if j3d and len(j3d) >= 1:
            com_3d_x = float(j3d[0][0])
            com_3d_y = float(j3d[0][1])
            com_3d_z = float(j3d[0][2])

        balance_batch.append(BalanceMetrics(
            frame_id=frame_obj.id,
            center_of_mass_x=com_x,
            center_of_mass_y=com_y,
            weight_distribution=weight_dist,
            stability_score=stability,
            center_of_mass_3d_x=com_3d_x,
            center_of_mass_3d_y=com_3d_y,
            center_of_mass_3d_z=com_3d_z,
        ))

        if len(angle_batch) >= BATCH_SIZE:
            session.add_all(angle_batch)
            session.add_all(balance_batch)
            session.flush()
            angle_batch = []
            balance_batch = []

    if angle_batch:
        session.add_all(angle_batch)
    if balance_batch:
        session.add_all(balance_batch)
    if angle_batch or balance_batch:
        session.flush()

    return len(all_frames)


def _analyze_dancer(performance_id, performance_dancer_id, label, frames_data, metadata, beat_data=None):
    """Run statistics, LLM coaching, scoring, and rhythm analysis for one dancer's frames."""
    pose_summary = summarize_pose_statistics(frames_data)

    # Rhythm scoring: correlate foot strikes with audio onsets
    rhythm_score = None
    rhythm_details = {}
    if beat_data and beat_data.get("onset_timestamps_ms"):
        fps = metadata.get("fps", 30.0)
        foot_strikes = detect_foot_strikes(frames_data, fps)
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
        frame_count=len(frames_data),
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

    try:
        # Stage 1: Ingest
        _update_progress(performance_id, "ingest", 5.0)
        metadata = extract_metadata(video_path)

        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.duration_ms = metadata["duration_ms"]

        total_frames = metadata["total_frames"]
        is_cancelled = _make_cancel_checker(performance_id)

        # Stage 1b: Audio beat detection (fast, CPU-only, runs before pose estimation)
        _update_progress(performance_id, "beat_detection", 10.0)
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
            # Multi-dancer mode
            track_id_to_info = {t["track_id"]: t for t in selected_tracks}
            selected_ids = set(track_id_to_info.keys())

            # Fetch detection bboxes to seed the tracker with known positions
            seed_bboxes = {}
            with get_session() as session:
                detected = session.query(DetectedPerson).filter(
                    DetectedPerson.performance_id == performance_id
                ).all()
                for dp in detected:
                    bbox = dp.bbox
                    seed_bboxes[dp.track_id] = (bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"])

            def pose_progress(current_frame: int, total: int):
                pct = 20.0 + (current_frame / max(total, 1)) * 60.0
                _update_progress(performance_id, "pose_estimation", pct, frame=current_frame, total_frames=total)

            _update_progress(performance_id, "pose_estimation", 20.0, frame=0, total_frames=total_frames)
            per_dancer_frames = run_pose_estimation_multi(
                video_path, metadata, selected_ids, progress_callback=pose_progress,
                is_cancelled=is_cancelled, seed_bboxes=seed_bboxes,
            )

            # Store frames per dancer
            _update_progress(performance_id, "pose_analysis", 80.0)
            total_stored = 0
            for track_id, frames_data in per_dancer_frames.items():
                info = track_id_to_info[track_id]
                pd_id = info["performance_dancer_id"]
                with get_session() as session:
                    total_stored += _store_frames_and_metrics(session, performance_id, pd_id, frames_data)

            # Analyze each dancer (runs on whatever frames were collected)
            num_dancers = len(selected_tracks)
            for i, (track_id, frames_data) in enumerate(per_dancer_frames.items()):
                info = track_id_to_info[track_id]
                pct = 83.0 + (i / max(num_dancers, 1)) * 12.0
                _update_progress(performance_id, "llm_synthesis", pct)
                if frames_data:
                    _analyze_dancer(
                        performance_id, info["performance_dancer_id"],
                        info.get("label"), frames_data, metadata, beat_data=beat_data,
                    )

        else:
            # Single-dancer mode (backward compatible)
            def pose_progress(current_frame: int, total: int):
                pct = 20.0 + (current_frame / max(total, 1)) * 63.0
                _update_progress(performance_id, "pose_estimation", pct, frame=current_frame, total_frames=total)

            _update_progress(performance_id, "pose_estimation", 20.0, frame=0, total_frames=total_frames)
            frames_data = run_pose_estimation(video_path, metadata, progress_callback=pose_progress, is_cancelled=is_cancelled)

            with get_session() as session:
                _store_frames_and_metrics(session, performance_id, None, frames_data)

            _update_progress(performance_id, "pose_analysis", 83.0)
            _update_progress(performance_id, "llm_synthesis", 85.0)
            _analyze_dancer(performance_id, None, None, frames_data, metadata, beat_data=beat_data)

        # Dispatch WHAM 3D estimation (fire-and-forget, runs in separate container)
        video_info = {"width": metadata.get("width", 1920), "height": metadata.get("height", 1080), "fps": metadata.get("fps", 30)}
        dispatch_wham_3d(performance_id, video_path, video_info)

        # Complete
        _update_progress(performance_id, "scoring", 95.0)
        _update_progress(performance_id, "complete", 100.0)

        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "complete"

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

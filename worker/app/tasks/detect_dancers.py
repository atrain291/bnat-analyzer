import logging
import subprocess
import traceback

from app.celery_app import app
from app.db import get_session
from app.models.performance import Performance, DetectedPerson
from app.pipeline.ingest import extract_metadata, ensure_browser_playable
from app.pipeline.pose import run_detection_pass
from app.pipeline.tracker import run_tracker
from app.tasks.video_pipeline import _make_progress_updater

logger = logging.getLogger(__name__)

DETECTION_FRAMES = 50


def _save_detection_frame(video_path: str, output_path: str):
    """Extract first frame as JPEG for the selection screen, using GPU decode if available."""
    # Probe codec to use matching CUVID decoder
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path,
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    codec = probe.stdout.strip() if probe.returncode == 0 else ""

    nvdec_codecs = {"h264": "h264_cuvid", "hevc": "hevc_cuvid", "vp9": "vp9_cuvid", "av1": "av1_cuvid"}
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if codec in nvdec_codecs:
        cmd += ["-hwaccel", "cuda", "-c:v", nvdec_codecs[codec]]
    cmd += ["-i", video_path, "-vframes", "1", "-q:v", "2", output_path]
    subprocess.run(cmd, check=True)


@app.task(bind=True, name="worker.app.tasks.detect_dancers.run_detection")
def run_detection(self, performance_id: int, video_path: str):
    logger.info(f"Starting detection for performance {performance_id}: {video_path}")
    update_progress = _make_progress_updater(performance_id, status="detecting")

    try:
        update_progress("ingest", 2.0, message="Checking video codec...")
        ensure_browser_playable(video_path, progress_callback=lambda msg: update_progress("ingest", 3.0, message=msg))
        update_progress("ingest", 5.0, message="Extracting video metadata...")
        metadata = extract_metadata(video_path)

        update_progress("ingest", 7.0, message="Saving preview frame...")
        frame_path = video_path.rsplit(".", 1)[0] + "_detection.jpg"
        _save_detection_frame(video_path, frame_path)
        video_key = video_path.split("/")[-1]
        detection_frame_key = video_key.rsplit(".", 1)[0] + "_detection.jpg"

        # Consolidated: update duration_ms + detection_frame_url in one session
        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.duration_ms = metadata["duration_ms"]
                perf.detection_frame_url = f"/uploads/{detection_frame_key}"

        # Run detection pass
        total_detect = min(DETECTION_FRAMES, metadata["total_frames"])

        def detection_progress(current: int, total: int):
            pct = 10.0 + (current / max(total, 1)) * 80.0
            update_progress("detection", pct, frame=current, total_frames=total,
                            message=f"Detecting persons... frame {current}/{total}")

        update_progress("detection", 10.0, frame=0, total_frames=total_detect,
                        message="Starting person detection...")
        all_frames = run_detection_pass(video_path, metadata, max_frames=DETECTION_FRAMES, progress_callback=detection_progress)

        update_progress("detection", 92.0, message="Assigning stable IDs with tracker...")
        persons = run_tracker(all_frames, min_frame_ratio=0.2)

        # Consolidated: store detected persons + set awaiting_selection in one session
        with get_session() as session:
            for person in persons:
                dp = DetectedPerson(
                    performance_id=performance_id,
                    track_id=person["track_id"],
                    bbox=person["bbox"],
                    representative_pose=person["representative_pose"],
                    frame_count=person["frame_count"],
                    area=person["area"],
                    appearance=person.get("appearance"),
                    color_histogram=person.get("color_histogram"),
                )
                session.add(dp)

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

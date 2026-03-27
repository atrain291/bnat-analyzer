import logging
import subprocess
from typing import Callable

import numpy as np
from rtmlib import Wholebody

from app.pipeline.pose_config import POSE_FRAME_SKIP, POSE_MAX_HEIGHT, POSE_USE_TENSORRT

logger = logging.getLogger(__name__)

# COCO-WholeBody 133 keypoints
# 0-16: Body (17), 17-22: Feet (6), 23-90: Face (68), 91-111: Left hand (21), 112-132: Right hand (21)

BODY_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

FOOT_KEYPOINT_NAMES = [
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

# Face keypoints (68) - stored as indexed array, not individually named
FACE_KEYPOINT_COUNT = 68

# Hand keypoints (21 per hand)
HAND_KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Combined body + feet keypoint names (the primary pose dict keys)
KEYPOINT_NAMES = BODY_KEYPOINT_NAMES + FOOT_KEYPOINT_NAMES

# Total keypoints in the wholebody model
TOTAL_KEYPOINTS = 133

# NVDEC codec mapping
NVDEC_CODECS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid",
}


def _build_ffmpeg_cmd(
    video_path: str, codec: str, width: int, height: int,
    frame_skip: int = 1, max_height: int = 0,
) -> tuple[list[str], int, int, int]:
    """Build ffmpeg command for frame extraction, with optional NVDEC acceleration.

    Returns (cmd, output_width, output_height, effective_total_frames_divisor).
    The caller must divide total_frames by the divisor for accurate progress.
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    nvdec = NVDEC_CODECS.get(codec)
    if nvdec:
        cmd += ["-hwaccel", "cuda", "-c:v", nvdec]

    cmd += ["-i", video_path]

    # Build video filters for frame skipping and resolution downscaling
    vf_parts = []

    if frame_skip > 1:
        vf_parts.append(f"select='not(mod(n\\,{frame_skip}))'")
        vf_parts.append("setpts=N/FRAME_RATE/TB")

    out_w, out_h = width, height
    if max_height > 0 and height > max_height:
        out_w = int(width * max_height / height)
        out_w = out_w + (out_w % 2)  # ensure even width
        out_h = max_height + (max_height % 2)  # ensure even height
        vf_parts.append(f"scale={out_w}:{out_h}")

    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "pipe:1",
    ]
    return cmd, out_w, out_h, frame_skip


def _init_model(device: str = "cuda") -> Wholebody:
    """Initialize RTMPose WholeBody model."""
    if POSE_USE_TENSORRT:
        from app.pipeline.tensorrt_setup import enable_tensorrt
        enable_tensorrt()
    return Wholebody(
        to_openpose=False,
        mode="performance",
        backend="onnxruntime",
        device=device,
    )


def _extract_single_person_pose(
    kp: np.ndarray,
    sc: np.ndarray,
    width: int,
    height: int,
) -> dict:
    """Convert keypoints/scores for one person into structured pose dict."""
    dancer_pose = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        dancer_pose[name] = {
            "x": round(float(kp[i][0]) / width, 5),
            "y": round(float(kp[i][1]) / height, 5),
            "z": 0.0,
            "confidence": round(float(sc[i]), 4),
        }

    left_hand = {}
    for i, name in enumerate(HAND_KEYPOINT_NAMES):
        idx = 91 + i
        left_hand[name] = {
            "x": round(float(kp[idx][0]) / width, 5),
            "y": round(float(kp[idx][1]) / height, 5),
            "confidence": round(float(sc[idx]), 4),
        }

    right_hand = {}
    for i, name in enumerate(HAND_KEYPOINT_NAMES):
        idx = 112 + i
        right_hand[name] = {
            "x": round(float(kp[idx][0]) / width, 5),
            "y": round(float(kp[idx][1]) / height, 5),
            "confidence": round(float(sc[idx]), 4),
        }

    face = []
    for i in range(FACE_KEYPOINT_COUNT):
        idx = 23 + i
        face.append({
            "x": round(float(kp[idx][0]) / width, 5),
            "y": round(float(kp[idx][1]) / height, 5),
            "confidence": round(float(sc[idx]), 4),
        })

    # Compute normalized bounding box from body keypoints
    valid_mask = sc[:23] > 0.3
    if valid_mask.sum() >= 3:
        valid_pts = kp[:23][valid_mask]
        bbox = (
            round(float(valid_pts[:, 0].min()) / width, 5),
            round(float(valid_pts[:, 1].min()) / height, 5),
            round(float(valid_pts[:, 0].max()) / width, 5),
            round(float(valid_pts[:, 1].max()) / height, 5),
        )
    else:
        bbox = (0.0, 0.0, 0.0, 0.0)

    return {
        "dancer_pose": dancer_pose,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "face": face,
        "bbox": bbox,
    }


def _extract_motion_state(pose: dict, bbox: tuple, prev_bbox: tuple | None = None) -> np.ndarray | None:
    """Extract compact 7-float pose motion state for movement correlation.

    Returns [avg_knee_angle, torso_angle, arm_ext_left, arm_ext_right,
             height_ratio, velocity_x, velocity_y] or None if insufficient data.
    Angles normalized to 0-1 range (divided by 180).
    """
    import math

    def _kp_angle(p1, p2, p3):
        if any(p.get("confidence", 0) < 0.3 for p in [p1, p2, p3]):
            return None
        v1 = (p1["x"] - p2["x"], p1["y"] - p2["y"])
        v2 = (p3["x"] - p2["x"], p3["y"] - p2["y"])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        m1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        m2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if m1 < 1e-6 or m2 < 1e-6:
            return None
        return math.degrees(math.acos(max(-1, min(1, dot / (m1 * m2)))))

    lk = _kp_angle(pose.get("left_hip", {}), pose.get("left_knee", {}), pose.get("left_ankle", {}))
    rk = _kp_angle(pose.get("right_hip", {}), pose.get("right_knee", {}), pose.get("right_ankle", {}))
    avg_knee = ((lk or 0) + (rk or 0)) / max(1, (1 if lk else 0) + (1 if rk else 0))
    if lk is None and rk is None:
        return None

    # Torso angle
    ls = pose.get("left_shoulder", {})
    rs = pose.get("right_shoulder", {})
    lh = pose.get("left_hip", {})
    rh = pose.get("right_hip", {})
    torso = 0.0
    if all(p.get("confidence", 0) > 0.3 for p in [ls, rs, lh, rh]):
        mx = (ls["x"] + rs["x"]) / 2 - (lh["x"] + rh["x"]) / 2
        my = (ls["y"] + rs["y"]) / 2 - (lh["y"] + rh["y"]) / 2
        if abs(my) > 1e-6:
            torso = abs(math.degrees(math.atan2(mx, -my)))

    # Arm extension
    la = _kp_angle(pose.get("left_shoulder", {}), pose.get("left_elbow", {}), pose.get("left_wrist", {}))
    ra = _kp_angle(pose.get("right_shoulder", {}), pose.get("right_elbow", {}), pose.get("right_wrist", {}))

    # Height ratio (bbox height vs width — captures crouching vs standing)
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    height_ratio = bh / bw if bw > 1e-6 else 1.5

    # Velocity
    vx, vy = 0.0, 0.0
    if prev_bbox is not None:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        pcx = (prev_bbox[0] + prev_bbox[2]) / 2
        pcy = (prev_bbox[1] + prev_bbox[3]) / 2
        vx = cx - pcx
        vy = cy - pcy

    return np.array([
        avg_knee / 180.0,
        torso / 180.0,
        (la or 90) / 180.0,
        (ra or 90) / 180.0,
        min(height_ratio / 3.0, 1.0),
        vx * 10,  # Scale up small velocities for better discrimination
        vy * 10,
    ], dtype=np.float32)


def _extract_all_poses(
    keypoints: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> list[dict]:
    """Extract pose data for ALL detected persons in a frame."""
    if keypoints is None or len(keypoints) == 0:
        return []

    results = []
    for idx in range(len(keypoints)):
        person = _extract_single_person_pose(keypoints[idx], scores[idx], width, height)
        # Skip persons with very few valid keypoints
        valid_count = sum(1 for name in KEYPOINT_NAMES if person["dancer_pose"][name]["confidence"] > 0.3)
        if valid_count >= 5:
            results.append(person)

    return results


def _extract_pose_data(
    keypoints: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> dict:
    """Convert RTMPose output arrays into structured pose dict.

    Args:
        keypoints: (N, 133, 2) array of keypoint coordinates.
        scores: (N, 133) array of confidence scores.
        width: Frame width for normalization.
        height: Frame height for normalization.

    Returns:
        Dict with dancer_pose, hands, and face data.
    """
    if keypoints is None or len(keypoints) == 0:
        return {"dancer_pose": {}, "left_hand": {}, "right_hand": {}, "face": []}

    # Pick the person with the largest bounding area from keypoint spread
    if len(keypoints) > 1:
        areas = []
        for kp in keypoints:
            valid = kp[kp[:, 0] > 0]
            if len(valid) > 0:
                x_range = valid[:, 0].max() - valid[:, 0].min()
                y_range = valid[:, 1].max() - valid[:, 1].min()
                areas.append(x_range * y_range)
            else:
                areas.append(0)
        best_idx = int(np.argmax(areas))
    else:
        best_idx = 0

    kp = keypoints[best_idx]  # (133, 2)
    sc = scores[best_idx]     # (133,)

    # Body + feet keypoints (indices 0-22) -> named dict
    dancer_pose = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        dancer_pose[name] = {
            "x": round(float(kp[i][0]) / width, 5),
            "y": round(float(kp[i][1]) / height, 5),
            "z": 0.0,
            "confidence": round(float(sc[i]), 4),
        }

    # Left hand keypoints (indices 91-111)
    left_hand = {}
    for i, name in enumerate(HAND_KEYPOINT_NAMES):
        idx = 91 + i
        left_hand[name] = {
            "x": round(float(kp[idx][0]) / width, 5),
            "y": round(float(kp[idx][1]) / height, 5),
            "confidence": round(float(sc[idx]), 4),
        }

    # Right hand keypoints (indices 112-132)
    right_hand = {}
    for i, name in enumerate(HAND_KEYPOINT_NAMES):
        idx = 112 + i
        right_hand[name] = {
            "x": round(float(kp[idx][0]) / width, 5),
            "y": round(float(kp[idx][1]) / height, 5),
            "confidence": round(float(sc[idx]), 4),
        }

    # Face keypoints (indices 23-90) - stored as list of {x, y, confidence}
    face = []
    for i in range(FACE_KEYPOINT_COUNT):
        idx = 23 + i
        face.append({
            "x": round(float(kp[idx][0]) / width, 5),
            "y": round(float(kp[idx][1]) / height, 5),
            "confidence": round(float(sc[idx]), 4),
        })

    return {
        "dancer_pose": dancer_pose,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "face": face,
    }


def run_pose_estimation(
    video_path: str,
    metadata: dict,
    progress_callback: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
):
    """Yield per-frame pose dicts from RTMPose WholeBody (133 keypoints).

    Yields dicts with dancer_pose, left_hand, right_hand, face, timestamp_ms.
    """

    model = _init_model()

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    codec = metadata.get("codec", "h264")
    total_frames = metadata["total_frames"]

    frame_skip = POSE_FRAME_SKIP
    max_height = POSE_MAX_HEIGHT
    cmd, w, h, skip = _build_ffmpeg_cmd(video_path, codec, width, height,
                                         frame_skip=frame_skip, max_height=max_height)
    frame_size = w * h * 3
    effective_total = total_frames // skip

    logger.info(f"Starting pose estimation (RTMPose WholeBody 133-pt): "
                f"{total_frames} frames at {fps:.1f} fps "
                f"(skip={skip}, scale={w}x{h}, effective={effective_total})")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_idx = 0

    try:
        while True:
            raw = process.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            keypoints, scores = model(frame)

            timestamp_ms = int((frame_idx * skip / fps) * 1000)

            pose_data = _extract_pose_data(keypoints, scores, w, h)
            pose_data["timestamp_ms"] = timestamp_ms
            yield pose_data

            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, effective_total)
            if is_cancelled and frame_idx % 10 == 0 and is_cancelled():
                logger.info(f"Pose estimation cancelled at frame {frame_idx}/{effective_total}")
                break

    finally:
        process.stdout.close()
        process.wait()

    logger.info(f"Pose estimation complete: {frame_idx} frames processed (of {effective_total} effective)")


def run_detection_pass(
    video_path: str,
    metadata: dict,
    max_frames: int = 50,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[list[dict]]:
    """Run pose estimation on the first N frames, returning ALL persons per frame.

    Returns a list of frames, each containing a list of person dicts.
    Each person dict includes 'appearance' and 'color_histogram' extracted from the frame.
    """
    from app.pipeline.appearance import extract_appearance

    model = _init_model()

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    codec = metadata.get("codec", "h264")
    total_frames = min(max_frames, metadata["total_frames"])

    cmd, w, h, _ = _build_ffmpeg_cmd(video_path, codec, width, height)
    frame_size = w * h * 3

    logger.info(f"Detection pass: scanning first {total_frames} frames for persons")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    all_frames = []
    frame_idx = 0

    try:
        while frame_idx < total_frames:
            raw = process.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            keypoints, scores = model(frame)

            persons = _extract_all_poses(keypoints, scores, w, h)

            # Extract appearance for each detected person
            for person in persons:
                try:
                    appearance = extract_appearance(frame, person["bbox"], normalized=True)
                    person["appearance"] = appearance
                    person["color_histogram"] = appearance.get("color_histogram", [])
                except Exception:
                    pass

            all_frames.append(persons)

            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)

    finally:
        process.stdout.close()
        process.wait()

    logger.info(f"Detection pass complete: {len(all_frames)} frames, max {max(len(f) for f in all_frames) if all_frames else 0} persons/frame")
    return all_frames


def run_pose_estimation_multi(
    video_path: str,
    metadata: dict,
    selected_track_ids: set[int],
    tracker_class=None,
    progress_callback: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    seed_bboxes: dict[int, tuple] | None = None,
    seed_histograms: dict[int, list[float]] | None = None,
):
    """Yield (track_id, frame_dict) tuples for selected persons.

    Args:
        seed_bboxes: {track_id: (x_min, y_min, x_max, y_max)} from detection pass.
            Seeds the tracker so it assigns the same IDs as the detection pass.
        seed_histograms: {track_id: [float, ...]} color histograms from detection pass.

    Yields (track_id, frame_data_dict) for selected tracks only.
    """
    from app.pipeline.tracker import SimpleTracker
    from app.pipeline.appearance import extract_appearance
    from app.pipeline.biometrics import extract_biometric_signature
    from app.pipeline.pose_config import REID_ENABLED

    reid_extractor = None
    if REID_ENABLED:
        try:
            from app.pipeline.reid import ReIDExtractor
            reid_extractor = ReIDExtractor()
            if reid_extractor.session is None:
                reid_extractor = None
        except Exception as e:
            logger.warning(f"Failed to initialize Re-ID extractor: {e}")

    model = _init_model()

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    codec = metadata.get("codec", "h264")
    total_frames = metadata["total_frames"]

    effective_fps = fps / POSE_FRAME_SKIP
    tracker = SimpleTracker(effective_fps=effective_fps)

    # Seed tracker with known positions, appearance, and group membership
    if seed_bboxes:
        tracker.seed(seed_bboxes, histograms=seed_histograms, group_ids=selected_track_ids)

    frame_skip = POSE_FRAME_SKIP
    max_height = POSE_MAX_HEIGHT
    cmd, w, h, skip = _build_ffmpeg_cmd(video_path, codec, width, height,
                                         frame_skip=frame_skip, max_height=max_height)
    frame_size = w * h * 3
    effective_total = total_frames // skip

    logger.info(f"Multi-person pose estimation: tracking {len(selected_track_ids)} persons "
                f"across {total_frames} frames (skip={skip}, scale={w}x{h}, effective={effective_total})")

    STALL_THRESHOLD_MS = 60_000  # abort if ALL dancers lost for 60s of video time
    RESEED_WAIT = 180  # frames (~3s at 60fps) without any selected dancer before re-seeding
    MAX_RESEEDS = 10

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_counts: dict[int, int] = {tid: 0 for tid in selected_track_ids}
    last_capture_ms: dict[int, int] = {tid: 0 for tid in selected_track_ids}  # per-dancer
    last_any_capture_ms = 0  # most recent video timestamp where any selected dancer was captured
    reseed_gap = 0  # consecutive frames with no selected dancer captured
    reseed_count = 0
    frame_idx = 0
    prev_bboxes: dict[int, tuple] = {}  # det_idx -> previous bbox for velocity

    try:
        while True:
            raw = process.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            keypoints, scores = model(frame)

            timestamp_ms = int((frame_idx * skip / fps) * 1000)
            persons = _extract_all_poses(keypoints, scores, w, h)

            bboxes = [p["bbox"] for p in persons] if persons else []

            # Extract appearance + identity signals every 5th output frame,
            # or every frame when dancers are being lost (reseed pending/gap growing)
            need_identity = (frame_idx % 5 == 0
                             or tracker._reseed_pending
                             or reseed_gap > 0)
            hists = None
            bio_sigs = None
            reid_embs = None
            if persons and need_identity:
                hists = []
                bio_sigs = []
                reid_embs = []
                for p in persons:
                    try:
                        app = extract_appearance(frame, p["bbox"], normalized=True)
                        hists.append(app.get("color_histogram", []))
                    except Exception:
                        hists.append(None)
                    bio_sigs.append(extract_biometric_signature(p.get("dancer_pose", {})))
                    if reid_extractor:
                        reid_embs.append(reid_extractor.extract(frame, p["bbox"], normalized=True))
                    else:
                        reid_embs.append(None)

            # Extract motion state every frame (lightweight — just angles + velocity)
            motion_states = None
            if persons:
                motion_states = []
                for di, p in enumerate(persons):
                    prev_bb = prev_bboxes.get(di)
                    motion_states.append(
                        _extract_motion_state(p.get("dancer_pose", {}), p["bbox"], prev_bb)
                    )
                    prev_bboxes[di] = p["bbox"]

            # Always call tracker.update() — even with empty bboxes — so it can
            # increment missing counts and manage occlusion recovery properly.
            track_ids = tracker.update(bboxes, histograms=hists,
                                       biometrics=bio_sigs, embeddings=reid_embs,
                                       motions=motion_states)

            captured_this_frame = False
            for det_idx, tid in enumerate(track_ids):
                if tid in selected_track_ids:
                    person = persons[det_idx]
                    person["timestamp_ms"] = timestamp_ms
                    frame_counts[tid] += 1
                    captured_this_frame = True
                    last_capture_ms[tid] = timestamp_ms
                    yield (tid, person)

            if captured_this_frame:
                last_any_capture_ms = timestamp_ms
                reseed_gap = 0
            else:
                reseed_gap += 1
                if reseed_gap >= RESEED_WAIT and reseed_count < MAX_RESEEDS and seed_bboxes:
                    logger.info(f"Frame {frame_idx}: all dancers lost for {reseed_gap} frames, "
                                f"re-seeding tracker (reseed {reseed_count + 1}/{MAX_RESEEDS})")
                    tracker.reseed()
                    reseed_count += 1
                    reseed_gap = 0

            # Stall detection: abort only if ALL dancers have been lost for
            # STALL_THRESHOLD_MS. Individual dancers being occluded is normal.
            if frame_idx % 100 == 0 and last_any_capture_ms > 0:
                all_lost_for = timestamp_ms - last_any_capture_ms
                if all_lost_for > STALL_THRESHOLD_MS:
                    # Log per-dancer status for debugging
                    for tid in selected_track_ids:
                        gap = (timestamp_ms - last_capture_ms[tid]) / 1000
                        logger.warning(f"  Track {tid}: last captured {gap:.0f}s ago "
                                       f"({frame_counts[tid]} total frames)")
                    logger.warning(f"All dancers lost for {all_lost_for / 1000:.0f}s "
                                   f"of video time. Aborting pose estimation.")
                    break

            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, effective_total)
            if is_cancelled and frame_idx % 10 == 0 and is_cancelled():
                logger.info(f"Multi-person pose estimation cancelled at frame {frame_idx}/{effective_total}")
                break

    finally:
        process.stdout.close()
        process.wait()

    for tid, count in frame_counts.items():
        logger.info(f"Track {tid}: {count} frames captured")

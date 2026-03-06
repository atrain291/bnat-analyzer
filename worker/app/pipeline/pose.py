import logging
import subprocess
from typing import Callable

import numpy as np
from rtmlib import Wholebody

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


def _build_ffmpeg_cmd(video_path: str, codec: str, width: int, height: int) -> list[str]:
    """Build ffmpeg command for frame extraction, with optional NVDEC acceleration."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    nvdec = NVDEC_CODECS.get(codec)
    if nvdec:
        cmd += ["-hwaccel", "cuda", "-c:v", nvdec]

    cmd += [
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "pipe:1",
    ]
    return cmd, width, height


def _init_model(device: str = "cuda") -> Wholebody:
    """Initialize RTMPose WholeBody model."""
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
) -> list[dict]:
    """Run RTMPose WholeBody on each frame and return normalized keypoints.

    Returns 133 keypoints per frame: 17 body + 6 feet + 68 face + 42 hands.
    """

    model = _init_model()

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    codec = metadata.get("codec", "h264")
    total_frames = metadata["total_frames"]

    cmd, w, h = _build_ffmpeg_cmd(video_path, codec, width, height)
    frame_size = w * h * 3

    logger.info(f"Starting pose estimation (RTMPose WholeBody 133-pt): {total_frames} frames at {fps:.1f} fps")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames_data = []
    frame_idx = 0

    try:
        while True:
            raw = process.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            keypoints, scores = model(frame)

            timestamp_ms = int((frame_idx / fps) * 1000)

            pose_data = _extract_pose_data(keypoints, scores, w, h)
            pose_data["timestamp_ms"] = timestamp_ms
            frames_data.append(pose_data)

            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)
            if is_cancelled and frame_idx % 10 == 0 and is_cancelled():
                logger.info(f"Pose estimation cancelled at frame {frame_idx}/{total_frames}")
                break

    finally:
        process.stdout.close()
        process.wait()

    logger.info(f"Pose estimation complete: {len(frames_data)} frames processed")
    return frames_data


def run_detection_pass(
    video_path: str,
    metadata: dict,
    max_frames: int = 50,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[list[dict]]:
    """Run pose estimation on the first N frames, returning ALL persons per frame.

    Returns a list of frames, each containing a list of person dicts.
    """
    model = _init_model()

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    codec = metadata.get("codec", "h264")
    total_frames = min(max_frames, metadata["total_frames"])

    cmd, w, h = _build_ffmpeg_cmd(video_path, codec, width, height)
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
) -> dict[int, list[dict]]:
    """Run pose estimation on all frames, tracking selected persons only.

    Returns {track_id: [frame_data_dicts]} for selected tracks only.
    """
    from app.pipeline.tracker import SimpleTracker

    model = _init_model()
    tracker = SimpleTracker()

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    codec = metadata.get("codec", "h264")
    total_frames = metadata["total_frames"]

    cmd, w, h = _build_ffmpeg_cmd(video_path, codec, width, height)
    frame_size = w * h * 3

    logger.info(f"Multi-person pose estimation: tracking {len(selected_track_ids)} persons across {total_frames} frames")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    results: dict[int, list[dict]] = {tid: [] for tid in selected_track_ids}
    frame_idx = 0

    try:
        while True:
            raw = process.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            keypoints, scores = model(frame)

            timestamp_ms = int((frame_idx / fps) * 1000)
            persons = _extract_all_poses(keypoints, scores, w, h)

            if persons:
                bboxes = [p["bbox"] for p in persons]
                track_ids = tracker.update(bboxes)

                for det_idx, tid in enumerate(track_ids):
                    if tid in selected_track_ids:
                        person = persons[det_idx]
                        person["timestamp_ms"] = timestamp_ms
                        results[tid].append(person)

            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)
            if is_cancelled and frame_idx % 10 == 0 and is_cancelled():
                logger.info(f"Multi-person pose estimation cancelled at frame {frame_idx}/{total_frames}")
                break

    finally:
        process.stdout.close()
        process.wait()

    for tid, frames in results.items():
        logger.info(f"Track {tid}: {len(frames)} frames captured")
    return results

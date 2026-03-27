import logging
import subprocess
from typing import Callable

import numpy as np
from rtmlib import Wholebody

from app.pipeline.pose_config import POSE_FRAME_SKIP, POSE_MAX_HEIGHT, POSE_USE_TENSORRT, SAM2_FRAME_SKIP

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



def run_pose_estimation_cropped(
    video_path: str,
    metadata: dict,
    dancer_bboxes: list[dict],
    start_ms: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
):
    """Yield per-frame pose dicts from RTMPose on pre-cropped dancer regions.

    Args:
        dancer_bboxes: List of {timestamp_ms, bbox, mask_iou} from tracking_frames.
            bbox is {x_min, y_min, x_max, y_max} normalized 0-1.
        start_ms: Video timestamp to start from.

    Yields dicts with dancer_pose, left_hand, right_hand, face, timestamp_ms, bbox.
    """
    model = _init_model()

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    codec = metadata.get("codec", "h264")

    # Build bbox lookup by timestamp
    bbox_lookup = {b["timestamp_ms"]: b for b in dancer_bboxes}
    timestamps = sorted(bbox_lookup.keys())
    total = len(timestamps)

    if total == 0:
        return

    # Build ffmpeg command with output-mode seeking (frame-accurate)
    start_sec = start_ms / 1000.0
    frame_skip = SAM2_FRAME_SKIP
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    nvdec = NVDEC_CODECS.get(codec)
    if nvdec:
        cmd += ["-hwaccel", "cuda", "-c:v", nvdec]

    cmd += ["-i", video_path]

    # Output-mode seeking: -ss AFTER -i for frame accuracy
    if start_ms > 0:
        cmd += ["-ss", str(start_sec)]

    vf_parts = []
    if frame_skip > 1:
        vf_parts.append(f"select='not(mod(n\\,{frame_skip}))'")
        vf_parts.append("setpts=N/FRAME_RATE/TB")
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-v", "error", "pipe:1"]

    frame_size = width * height * 3
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_idx = 0
    BBOX_PAD = 0.25  # 25% padding around bbox

    logger.info(f"Cropped pose estimation: {total} tracking bboxes, "
                f"start={start_ms}ms, skip={frame_skip}")

    try:
        while True:
            raw = process.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break

            timestamp_ms = start_ms + int(frame_idx * frame_skip / fps * 1000)

            # Find tracking bbox for this timestamp
            entry = bbox_lookup.get(timestamp_ms)
            if entry is None:
                # Try nearest timestamp within 50ms
                nearest = min(timestamps, key=lambda t: abs(t - timestamp_ms), default=None)
                if nearest is not None and abs(nearest - timestamp_ms) < 50:
                    entry = bbox_lookup[nearest]

            frame_idx += 1

            if entry is None or entry.get("mask_iou", 0) < 0.3:
                continue

            bbox = entry["bbox"]
            bw = bbox["x_max"] - bbox["x_min"]
            bh = bbox["y_max"] - bbox["y_min"]
            if bw < 0.01 or bh < 0.01:
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            # Denormalize bbox and add padding
            pad_x = bw * BBOX_PAD
            pad_y = bh * BBOX_PAD
            px1 = max(0, int((bbox["x_min"] - pad_x) * width))
            py1 = max(0, int((bbox["y_min"] - pad_y) * height))
            px2 = min(width, int((bbox["x_max"] + pad_x) * width))
            py2 = min(height, int((bbox["y_max"] + pad_y) * height))

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            crop_h, crop_w = crop.shape[:2]

            # Run RTMPose on the crop
            keypoints, scores = model(crop)
            if keypoints is None or len(keypoints) == 0:
                continue

            # Extract single person pose data (normalized to crop dimensions)
            pose_data = _extract_pose_data(keypoints, scores, crop_w, crop_h)

            # Transform crop-normalized coords to full-frame normalized coords
            for kp_name, kp in pose_data.get("dancer_pose", {}).items():
                kp["x"] = kp["x"] * (crop_w / width) + (px1 / width)
                kp["y"] = kp["y"] * (crop_h / height) + (py1 / height)

            for hand_key in ("left_hand", "right_hand"):
                for kp_name, kp in pose_data.get(hand_key, {}).items():
                    kp["x"] = kp["x"] * (crop_w / width) + (px1 / width)
                    kp["y"] = kp["y"] * (crop_h / height) + (py1 / height)

            for kp in pose_data.get("face", []):
                kp["x"] = kp["x"] * (crop_w / width) + (px1 / width)
                kp["y"] = kp["y"] * (crop_h / height) + (py1 / height)

            pose_data["bbox"] = (
                round(bbox["x_min"], 5), round(bbox["y_min"], 5),
                round(bbox["x_max"], 5), round(bbox["y_max"], 5),
            )
            pose_data["timestamp_ms"] = timestamp_ms
            yield pose_data

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total)
            if is_cancelled and frame_idx % 10 == 0 and is_cancelled():
                break

    finally:
        process.stdout.close()
        process.wait()

    logger.info(f"Cropped pose estimation complete: {frame_idx} frames processed")

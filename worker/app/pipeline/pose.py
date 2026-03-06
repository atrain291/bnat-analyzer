import logging
import subprocess
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO keypoint names in order
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

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


def run_pose_estimation(
    video_path: str,
    metadata: dict,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Run YOLOv8-Pose on each frame and return normalized keypoints."""

    model = YOLO("yolov8n-pose.pt")

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    codec = metadata.get("codec", "h264")
    total_frames = metadata["total_frames"]

    cmd, w, h = _build_ffmpeg_cmd(video_path, codec, width, height)
    frame_size = w * h * 3

    logger.info(f"Starting pose estimation: {total_frames} frames at {fps:.1f} fps")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames_data = []
    frame_idx = 0

    try:
        while True:
            raw = process.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            results = model(frame, verbose=False)

            timestamp_ms = int((frame_idx / fps) * 1000)

            # Get the best detection (highest confidence person)
            dancer_pose = {}
            if results and len(results[0].keypoints) > 0:
                # Pick the person with the largest bounding box (likely the main dancer)
                best_idx = 0
                if len(results[0].boxes) > 1:
                    areas = []
                    for box in results[0].boxes.xyxy:
                        x1, y1, x2, y2 = box.cpu().numpy()
                        areas.append((x2 - x1) * (y2 - y1))
                    best_idx = int(np.argmax(areas))

                kps = results[0].keypoints[best_idx]
                xy = kps.xy[0].cpu().numpy()
                conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(17)

                for i, name in enumerate(KEYPOINT_NAMES):
                    dancer_pose[name] = {
                        "x": round(float(xy[i][0]) / w, 5),
                        "y": round(float(xy[i][1]) / h, 5),
                        "z": 0.0,
                        "confidence": round(float(conf[i]), 4),
                    }

            frames_data.append({
                "timestamp_ms": timestamp_ms,
                "dancer_pose": dancer_pose,
            })

            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)

    finally:
        process.stdout.close()
        process.wait()

    logger.info(f"Pose estimation complete: {len(frames_data)} frames processed")
    return frames_data

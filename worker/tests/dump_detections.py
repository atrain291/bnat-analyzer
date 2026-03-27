"""Dump RTMPose detections for the first N frames to JSON for offline tracker testing.

Run inside the worker container:
    python -m tests.dump_detections /app/uploads/VIDEO.mp4 --frames 500

Output: /app/uploads/detections_dump.json
"""
import argparse
import json
import logging
import subprocess
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--frames", type=int, default=500, help="Max effective frames to dump")
    parser.add_argument("--output", default="/app/uploads/detections_dump.json")
    args = parser.parse_args()

    from app.pipeline.ingest import extract_metadata
    from app.pipeline.pose import _init_model, _extract_all_poses, _build_ffmpeg_cmd, POSE_FRAME_SKIP, POSE_MAX_HEIGHT
    from app.pipeline.appearance import extract_appearance
    from app.pipeline.biometrics import extract_biometric_signature

    metadata = extract_metadata(args.video_path)
    logger.info(f"Video: {metadata['width']}x{metadata['height']} @ {metadata['fps']:.1f}fps, "
                f"{metadata['total_frames']} frames, codec={metadata.get('codec')}")

    model = _init_model()
    fps = metadata["fps"]
    width, height = metadata["width"], metadata["height"]
    codec = metadata.get("codec", "h264")

    cmd, w, h, skip = _build_ffmpeg_cmd(args.video_path, codec, width, height,
                                         frame_skip=POSE_FRAME_SKIP, max_height=POSE_MAX_HEIGHT)
    frame_size = w * h * 3
    max_frames = min(args.frames, metadata["total_frames"] // skip)

    logger.info(f"Extracting {max_frames} effective frames (skip={skip}, scale={w}x{h})")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    all_frames = []

    for frame_idx in range(max_frames):
        raw = process.stdout.read(frame_size)
        if not raw or len(raw) < frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        keypoints, scores = model(frame)

        timestamp_ms = int((frame_idx * skip / fps) * 1000)
        persons = _extract_all_poses(keypoints, scores, w, h)

        frame_data = {
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "persons": [],
        }
        for p in persons:
            person = {
                "bbox": list(p["bbox"]),
                "dancer_pose": {k: v for k, v in p.get("dancer_pose", {}).items()},
            }
            # Extract appearance
            try:
                app = extract_appearance(frame, p["bbox"], normalized=True)
                person["color_histogram"] = app.get("color_histogram", [])
            except Exception:
                person["color_histogram"] = []
            frame_data["persons"].append(person)

        all_frames.append(frame_data)

        if frame_idx % 50 == 0:
            logger.info(f"Frame {frame_idx}/{max_frames} ({timestamp_ms}ms) — {len(persons)} persons")

    process.stdout.close()
    process.wait()

    with open(args.output, "w") as f:
        json.dump({"metadata": metadata, "frames": all_frames}, f)

    logger.info(f"Dumped {len(all_frames)} frames to {args.output}")
    logger.info(f"Persons per frame: min={min(len(f['persons']) for f in all_frames)}, "
                f"max={max(len(f['persons']) for f in all_frames)}, "
                f"avg={sum(len(f['persons']) for f in all_frames) / len(all_frames):.1f}")


if __name__ == "__main__":
    main()

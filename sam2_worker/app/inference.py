"""SAM 2.1 video segmentation inference."""

import logging
import subprocess
import json

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT = "/app/models/sam2.1_hiera_base_plus.pt"
STORE_EVERY_N = 2  # Store results every N frames (aligns with POSE_FRAME_SKIP)


def _get_video_info(video_path: str) -> dict:
    """Get video fps, width, height via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    fps_parts = stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    return {
        "fps": fps,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "total_frames": int(stream.get("nb_frames", 0)),
    }


def run_sam2_tracking(
    video_path: str,
    start_timestamp_ms: int,
    click_prompts: list[dict],
    is_cancelled=None,
    progress_callback=None,
) -> list[dict]:
    """Run SAM 2 video segmentation from click prompts.

    Args:
        video_path: Path to video file (mp4).
        start_timestamp_ms: Timestamp to start tracking from.
        click_prompts: List of {x, y, label} dicts (normalized 0-1 coords).
        is_cancelled: Callable returning True if task should stop.
        progress_callback: Callable(current_frame, total_frames).

    Returns:
        List of {dancer_index, timestamp_ms, bbox, mask_iou} dicts.
    """
    from sam2.build_sam import build_sam2_video_predictor

    video_info = _get_video_info(video_path)
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    total_frames = video_info["total_frames"]

    start_frame_idx = int(start_timestamp_ms / 1000.0 * fps)
    effective_total = total_frames - start_frame_idx

    logger.info(
        "SAM 2 tracking: %d dancers, start_frame=%d, total_frames=%d, "
        "video=%dx%d@%.1ffps",
        len(click_prompts), start_frame_idx, total_frames, width, height, fps,
    )

    # Load model and init video state
    predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device="cuda")
    state = predictor.init_state(video_path=video_path)

    # Add click prompts at the start frame
    # Points must be in ORIGINAL frame pixel coordinates (SAM 2 handles internal rescaling)
    for i, prompt in enumerate(click_prompts):
        points = np.array(
            [[prompt["x"] * width, prompt["y"] * height]], dtype=np.float32,
        )
        labels = np.array([1], dtype=np.int32)  # 1 = positive (foreground)
        _, _, _ = predictor.add_new_points_or_box(
            state, frame_idx=start_frame_idx, obj_id=i,
            points=points, labels=labels,
        )
        logger.info(
            "Added prompt %d at pixel (%.0f, %.0f)", i, points[0][0], points[0][1],
        )

    # Propagate forward through the video
    # propagate_in_video yields (frame_idx, obj_ids, masks_dict)
    # where masks_dict is {obj_id: mask_tensor}
    results = []
    frame_count = 0
    for frame_idx, obj_ids, masks_dict in predictor.propagate_in_video(
        state, start_frame_idx=start_frame_idx,
    ):
        # Only store every STORE_EVERY_N frames
        if (frame_idx - start_frame_idx) % STORE_EVERY_N != 0:
            continue

        timestamp_ms = int(frame_idx / fps * 1000)

        for obj_id in obj_ids:
            mask = masks_dict[obj_id]
            # mask shape: (1, H, W) -- squeeze to (H, W)
            mask_np = mask.squeeze().cpu().numpy() > 0.5

            # Derive bbox from mask
            ys, xs = np.where(mask_np)
            if len(ys) == 0:
                # Dancer not visible -- store zero bbox
                results.append({
                    "dancer_index": int(obj_id),
                    "timestamp_ms": timestamp_ms,
                    "bbox": {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0},
                    "mask_iou": 0.0,
                })
                continue

            x_min = float(xs.min()) / width
            y_min = float(ys.min()) / height
            x_max = float(xs.max()) / width
            y_max = float(ys.max()) / height

            # mask_iou: fraction of bbox covered by mask (density proxy for confidence)
            bbox_area = max(1, (xs.max() - xs.min()) * (ys.max() - ys.min()))
            mask_density = float(len(ys)) / bbox_area

            results.append({
                "dancer_index": int(obj_id),
                "timestamp_ms": timestamp_ms,
                "bbox": {
                    "x_min": round(x_min, 5), "y_min": round(y_min, 5),
                    "x_max": round(x_max, 5), "y_max": round(y_max, 5),
                },
                "mask_iou": round(mask_density, 4),
            })

        frame_count += 1
        if progress_callback and frame_count % 50 == 0:
            progress_callback(frame_idx - start_frame_idx, effective_total)

        if is_cancelled and frame_count % 50 == 0 and is_cancelled():
            logger.info("SAM 2 tracking cancelled at frame %d", frame_idx)
            break

    # Release GPU memory
    del predictor, state
    torch.cuda.empty_cache()

    logger.info(
        "SAM 2 tracking complete: %d frames processed, %d tracking entries",
        frame_count, len(results),
    )
    return results

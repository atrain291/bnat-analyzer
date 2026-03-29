"""SAM 2.1 video segmentation inference."""

import logging
import os
import shutil
import subprocess
import json
import tempfile

import gc

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT = "/app/models/sam2.1_hiera_base_plus.pt"
STORE_EVERY_N = 2  # Store results every N frames (aligns with POSE_FRAME_SKIP)
SAM2_MAX_FPS = int(os.environ.get("SAM2_MAX_FPS", "30"))
# Max frames to load into SAM 2 at once (1024×1024×3×4 bytes each ≈ 12MB/frame)
SAM2_CHUNK_FRAMES = int(os.environ.get("SAM2_CHUNK_FRAMES", "750"))
SAM2_OVERLAP_FRAMES = int(os.environ.get("SAM2_OVERLAP_FRAMES", "15"))
SAM2_COMPILE = os.environ.get("SAM2_COMPILE", "true").lower() in ("true", "1", "yes")


def _build_predictor():
    """Build SAM 2 video predictor with optional torch.compile."""
    from sam2.build_sam import build_sam2_video_predictor

    compile_overrides = []
    if SAM2_COMPILE:
        compile_overrides.append("++model.compile_image_encoder=true")
        logger.info("SAM 2 torch.compile enabled (first run will be slow)")

    predictor = build_sam2_video_predictor(
        MODEL_CFG, CHECKPOINT, device="cuda",
        hydra_overrides_extra=compile_overrides,
    )

    return predictor


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


def _extract_frames_to_dir(video_path: str, output_dir: str, target_fps: float,
                           start_ms: int = 0) -> int:
    """Extract video frames as JPEGs at a target FPS.

    Returns the number of frames extracted.
    """
    cmd = ["ffmpeg", "-y", "-v", "warning"]
    if start_ms > 0:
        cmd += ["-ss", f"{start_ms / 1000.0:.3f}"]
    cmd += [
        "-i", video_path,
        "-vf", f"fps={target_fps}",
        "-q:v", "2",
        os.path.join(output_dir, "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    count = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    return count


def _extract_bbox(mask_np, width, height):
    """Extract normalized bbox and mask density from a binary mask."""
    ys, xs = np.where(mask_np)
    if len(ys) == 0:
        return {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0}, 0.0

    x_min = float(xs.min()) / width
    y_min = float(ys.min()) / height
    x_max = float(xs.max()) / width
    y_max = float(ys.max()) / height
    bbox_area = max(1, (xs.max() - xs.min()) * (ys.max() - ys.min()))
    mask_density = float(len(ys)) / bbox_area

    return {
        "x_min": round(x_min, 5), "y_min": round(y_min, 5),
        "x_max": round(x_max, 5), "y_max": round(y_max, 5),
    }, round(mask_density, 4)


def _propagate_chunk(predictor, chunk_dir, click_prompts, width, height,
                     chunk_offset_ms, sam2_fps, is_cancelled, progress_callback,
                     global_frame_offset, global_total,
                     seed_masks=None, overlap_frames=0):
    """Run SAM 2 on a chunk of frames.

    Args:
        seed_masks: If provided, dict mapping obj_id -> np.ndarray (H×W bool)
            from the previous chunk's last valid masks. Used instead of click
            prompts to seed tracking continuity across chunks.
        overlap_frames: Number of leading frames to skip when collecting results
            (they were already captured by the previous chunk).

    Returns:
        (results, frame_count, end_masks) where end_masks maps obj_id -> last
        valid np.ndarray mask seen in this chunk.
    """
    state = predictor.init_state(
        video_path=chunk_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    # Seed tracking at frame 0
    if seed_masks is None:
        # First chunk: use original click prompts
        for i, prompt in enumerate(click_prompts):
            points = np.array(
                [[prompt["x"] * width, prompt["y"] * height]], dtype=np.float32,
            )
            labels = np.array([1], dtype=np.int32)
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=i,
                points=points, labels=labels,
            )
    else:
        # Subsequent chunks: seed from last known masks
        for i, prompt in enumerate(click_prompts):
            mask_np = seed_masks.get(i)
            if mask_np is not None and mask_np.any():
                mask_tensor = torch.from_numpy(mask_np).float()
                predictor.add_new_mask(state, frame_idx=0, obj_id=i, mask=mask_tensor)
            else:
                # Never had a valid mask — fall back to original click
                points = np.array(
                    [[prompt["x"] * width, prompt["y"] * height]], dtype=np.float32,
                )
                labels = np.array([1], dtype=np.int32)
                predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=i,
                    points=points, labels=labels,
                )
                logger.warning("obj_id %d has no carry mask, falling back to click prompt", i)

    results = []
    end_masks: dict[int, np.ndarray] = {}
    frame_count = 0
    for frame_idx, obj_ids, masks_dict in predictor.propagate_in_video(
        state, start_frame_idx=0,
    ):
        # Compute masks once per obj and track the latest valid mask for carryover
        frame_masks: dict[int, np.ndarray] = {}
        for obj_id in obj_ids:
            mask = masks_dict[obj_id]
            mask_np = mask.squeeze().cpu().numpy() > 0.5
            frame_masks[int(obj_id)] = mask_np
            if mask_np.any():
                end_masks[int(obj_id)] = mask_np

        if frame_idx % STORE_EVERY_N != 0:
            continue

        # Skip overlap frames (already captured by previous chunk)
        if frame_idx < overlap_frames:
            continue

        timestamp_ms = chunk_offset_ms + int(frame_idx / sam2_fps * 1000)

        for obj_id_int, mask_np in frame_masks.items():
            bbox, mask_iou = _extract_bbox(mask_np, width, height)
            results.append({
                "dancer_index": obj_id_int,
                "timestamp_ms": timestamp_ms,
                "bbox": bbox,
                "mask_iou": mask_iou,
            })

        frame_count += 1
        global_pos = global_frame_offset + frame_idx
        if progress_callback and frame_count % 50 == 0:
            progress_callback(global_pos, global_total)

        if is_cancelled and frame_count % 50 == 0 and is_cancelled():
            logger.info("SAM 2 tracking cancelled at frame %d (global %d)", frame_idx, global_pos)
            break

    predictor.reset_state(state)
    return results, frame_count, end_masks


def run_sam2_tracking(
    video_path: str,
    start_timestamp_ms: int,
    click_prompts: list[dict],
    is_cancelled=None,
    progress_callback=None,
) -> list[dict]:
    """Run SAM 2 video segmentation from click prompts.

    For long/high-fps videos, extracts frames at SAM2_MAX_FPS and processes
    in chunks of SAM2_CHUNK_FRAMES to stay within memory limits.
    """
    video_info = _get_video_info(video_path)
    orig_fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    total_frames = video_info["total_frames"]

    sam2_fps = min(orig_fps, SAM2_MAX_FPS)
    needs_extraction = orig_fps > SAM2_MAX_FPS

    # Calculate how many frames SAM 2 will process
    duration_s = total_frames / orig_fps
    start_s = start_timestamp_ms / 1000.0
    effective_duration = duration_s - start_s
    effective_frames = int(effective_duration * sam2_fps)

    needs_chunking = effective_frames > SAM2_CHUNK_FRAMES

    logger.info(
        "SAM 2 tracking: %d dancers, video=%dx%d@%.1ffps, duration=%.1fs, "
        "sam2_fps=%d, effective_frames=%d, chunk_size=%d, chunking=%s",
        len(click_prompts), width, height, orig_fps, duration_s,
        sam2_fps, effective_frames, SAM2_CHUNK_FRAMES, needs_chunking,
    )

    if not needs_extraction and not needs_chunking:
        # Small video at acceptable fps — use direct video path (original fast path)
        return _run_direct(
            video_path, start_timestamp_ms, click_prompts,
            orig_fps, width, height, total_frames,
            is_cancelled, progress_callback,
        )

    # Extract frames to temp dir, then process in chunks
    frame_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    try:
        num_extracted = _extract_frames_to_dir(
            video_path, frame_dir, sam2_fps, start_ms=start_timestamp_ms,
        )
        logger.info("Extracted %d frames at %dfps to %s", num_extracted, sam2_fps, frame_dir)

        if num_extracted == 0:
            logger.warning("No frames extracted")
            return []

        if num_extracted <= SAM2_CHUNK_FRAMES:
            # Fits in one chunk — process the whole dir
            return _run_direct_on_dir(
                frame_dir, start_timestamp_ms, click_prompts,
                sam2_fps, width, height, num_extracted,
                is_cancelled, progress_callback,
            )

        # Process in chunks
        return _run_chunked(
            frame_dir, start_timestamp_ms, click_prompts,
            sam2_fps, width, height, num_extracted,
            is_cancelled, progress_callback,
        )
    finally:
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir, ignore_errors=True)
            logger.info("Cleaned up frame directory %s", frame_dir)


def _run_direct(video_path, start_timestamp_ms, click_prompts,
                fps, width, height, total_frames, is_cancelled, progress_callback):
    """Original direct-video path for short, low-fps videos."""
    start_frame_idx = int(start_timestamp_ms / 1000.0 * fps)
    effective_total = total_frames - start_frame_idx

    predictor = _build_predictor()
    state = predictor.init_state(
        video_path=video_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    for i, prompt in enumerate(click_prompts):
        points = np.array(
            [[prompt["x"] * width, prompt["y"] * height]], dtype=np.float32,
        )
        labels = np.array([1], dtype=np.int32)
        predictor.add_new_points_or_box(
            state, frame_idx=start_frame_idx, obj_id=i,
            points=points, labels=labels,
        )
        logger.info("Added prompt %d at pixel (%.0f, %.0f)", i, points[0][0], points[0][1])

    results = []
    frame_count = 0
    for frame_idx, obj_ids, masks_dict in predictor.propagate_in_video(
        state, start_frame_idx=start_frame_idx,
    ):
        if (frame_idx - start_frame_idx) % STORE_EVERY_N != 0:
            continue

        timestamp_ms = int(frame_idx / fps * 1000)
        for obj_id in obj_ids:
            mask = masks_dict[obj_id]
            mask_np = mask.squeeze().cpu().numpy() > 0.5
            bbox, mask_iou = _extract_bbox(mask_np, width, height)
            results.append({
                "dancer_index": int(obj_id),
                "timestamp_ms": timestamp_ms,
                "bbox": bbox,
                "mask_iou": mask_iou,
            })

        frame_count += 1
        if progress_callback and frame_count % 50 == 0:
            progress_callback(frame_idx - start_frame_idx, effective_total)
        if is_cancelled and frame_count % 50 == 0 and is_cancelled():
            logger.info("SAM 2 tracking cancelled at frame %d", frame_idx)
            break

    predictor.reset_state(state)
    del predictor, state
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("SAM 2 tracking complete: %d frames, %d entries", frame_count, len(results))
    return results


def _run_direct_on_dir(frame_dir, start_timestamp_ms, click_prompts,
                       sam2_fps, width, height, num_frames,
                       is_cancelled, progress_callback):
    """Process an extracted frame directory that fits in one go."""
    predictor = _build_predictor()
    results, frame_count, _ = _propagate_chunk(
        predictor, frame_dir, click_prompts, width, height,
        start_timestamp_ms, sam2_fps, is_cancelled, progress_callback,
        global_frame_offset=0, global_total=num_frames,
    )

    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("SAM 2 tracking complete: %d frames, %d entries", frame_count, len(results))
    return results


def _run_chunked(frame_dir, start_timestamp_ms, click_prompts,
                 sam2_fps, width, height, num_frames,
                 is_cancelled, progress_callback):
    """Process extracted frames in chunks with mask carryover between chunks.

    Each chunk after the first is seeded with the last valid mask from the
    previous chunk (via add_new_mask), plus a small overlap window so SAM 2
    can re-establish temporal context before we start collecting results.
    """
    all_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    num_chunks = (len(all_files) + SAM2_CHUNK_FRAMES - 1) // SAM2_CHUNK_FRAMES
    logger.info("Processing %d frames in %d chunks of %d (overlap=%d)",
                len(all_files), num_chunks, SAM2_CHUNK_FRAMES, SAM2_OVERLAP_FRAMES)

    predictor = _build_predictor()
    all_results = []
    total_frame_count = 0

    # Mask carryover state between chunks
    carry_masks: dict[int, np.ndarray] = {}
    best_ever_masks: dict[int, np.ndarray] = {}

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * SAM2_CHUNK_FRAMES

        # For chunks after the first, include overlap from the prior chunk's tail
        if chunk_idx == 0:
            overlap = 0
            file_start = chunk_start
        else:
            overlap = min(SAM2_OVERLAP_FRAMES, chunk_start)
            file_start = chunk_start - overlap

        chunk_files = all_files[file_start:chunk_start + SAM2_CHUNK_FRAMES]
        if not chunk_files:
            break

        # Cap overlap to actual chunk size so at least 1 frame produces results
        overlap = min(overlap, max(0, len(chunk_files) - 1))

        # Build seed masks: prefer carry from previous chunk, fall back to best ever
        if chunk_idx == 0:
            seed_masks = None
        else:
            seed_masks = {}
            for i in range(len(click_prompts)):
                if i in carry_masks and carry_masks[i].any():
                    seed_masks[i] = carry_masks[i]
                elif i in best_ever_masks:
                    seed_masks[i] = best_ever_masks[i]
                    logger.warning("Chunk %d: obj %d using best_ever fallback mask", chunk_idx, i)
                # else: absent from seed_masks → _propagate_chunk falls back to click prompt

        chunk_dir = tempfile.mkdtemp(prefix=f"sam2_chunk{chunk_idx}_")
        try:
            for new_idx, fname in enumerate(chunk_files, start=1):
                src = os.path.join(frame_dir, fname)
                dst = os.path.join(chunk_dir, f"{new_idx:06d}.jpg")
                os.symlink(src, dst)

            # Offset is relative to the file_start (includes overlap)
            chunk_offset_ms = start_timestamp_ms + int(file_start / sam2_fps * 1000)

            logger.info(
                "Chunk %d/%d: frames %d-%d (overlap=%d), offset=%dms",
                chunk_idx + 1, num_chunks,
                file_start, file_start + len(chunk_files) - 1,
                overlap, chunk_offset_ms,
            )

            results, frame_count, end_masks = _propagate_chunk(
                predictor, chunk_dir, click_prompts, width, height,
                chunk_offset_ms, sam2_fps, is_cancelled, progress_callback,
                global_frame_offset=file_start, global_total=num_frames,
                seed_masks=seed_masks, overlap_frames=overlap,
            )
            all_results.extend(results)
            total_frame_count += frame_count

            # Update carryover state
            carry_masks = end_masks
            for obj_id, mask in end_masks.items():
                if mask.any():
                    best_ever_masks[obj_id] = mask

            if is_cancelled and is_cancelled():
                break

        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("SAM 2 tracking complete: %d frames, %d entries across %d chunks",
                total_frame_count, len(all_results), num_chunks)
    return all_results

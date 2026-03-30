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



PATCH_RADIUS = 8  # pixels around each keypoint to sample

# Keypoints on torso/limbs — the most stable for appearance (not wrists/ankles which move fast)
_APPEARANCE_KEYPOINTS = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_elbow", "right_elbow", "left_knee", "right_knee",
    "nose",
]


def _extract_joint_appearance(
    pose: dict,
    frame: np.ndarray,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    """Sample a small color patch around each keypoint from the frame.

    Returns {keypoint_name: mean_bgr_array} for confident keypoints.
    The mean color of the patch is used, not the raw pixels — this is
    robust to slight position jitter while capturing the actual appearance.
    """
    patches = {}
    r = PATCH_RADIUS
    for name in _APPEARANCE_KEYPOINTS:
        kp = pose.get(name)
        if not kp or kp.get("confidence", 0) < 0.3:
            continue
        px = int(kp["x"] * width)
        py = int(kp["y"] * height)
        y1 = max(0, py - r)
        y2 = min(height, py + r)
        x1 = max(0, px - r)
        x2 = min(width, px + r)
        if y2 - y1 < 3 or x2 - x1 < 3:
            continue
        patch = frame[y1:y2, x1:x2]
        patches[name] = patch.mean(axis=(0, 1)).astype(np.float32)  # mean BGR
    return patches


def _appearance_similarity(
    prev_patches: dict[str, np.ndarray],
    curr_patches: dict[str, np.ndarray],
) -> tuple[float, int]:
    """Compare appearance patches between two skeletons.

    Returns (mean_color_distance, matched_count).
    Color distance is L2 in BGR space (0-441 range), normalized to 0-1.
    Lower means more similar appearance.
    """
    MAX_COLOR_DIST = 441.67  # sqrt(255^2 * 3)
    total = 0.0
    matched = 0
    for name in _APPEARANCE_KEYPOINTS:
        if name in prev_patches and name in curr_patches:
            diff = prev_patches[name] - curr_patches[name]
            dist = float(np.sqrt(np.sum(diff * diff)))
            total += dist / MAX_COLOR_DIST
            matched += 1
    if matched == 0:
        return 1.0, 0
    return total / matched, matched


def _skeleton_identity_score(
    prev_pose: dict,
    prev_appearance: dict[str, np.ndarray],
    candidate_pose: dict,
    candidate_appearance: dict[str, np.ndarray],
) -> tuple[float, int]:
    """Score how likely a candidate skeleton is the same person as the previous frame.

    Combines:
    - Geometric continuity: average per-joint displacement (0-1 range)
    - Appearance continuity: average per-joint color similarity (0-1 range)

    Returns (combined_score, matched_joints). Lower score = better match.
    Appearance is weighted more heavily because geometry can be ambiguous
    when dancers are close, but clothing color is stable.
    """
    # Geometry: per-joint displacement
    geo_total = 0.0
    geo_matched = 0
    for name in BODY_KEYPOINT_NAMES + FOOT_KEYPOINT_NAMES:
        prev_kp = prev_pose.get(name)
        cand_kp = candidate_pose.get(name)
        if (prev_kp and cand_kp
                and prev_kp.get("confidence", 0) > 0.3
                and cand_kp.get("confidence", 0) > 0.3):
            dx = prev_kp["x"] - cand_kp["x"]
            dy = prev_kp["y"] - cand_kp["y"]
            geo_total += (dx * dx + dy * dy) ** 0.5
            geo_matched += 1

    if geo_matched == 0:
        return float("inf"), 0

    geo_score = geo_total / geo_matched  # mean displacement, ~0.01-0.10 range

    # Appearance: per-joint color distance
    app_score, app_matched = _appearance_similarity(prev_appearance, candidate_appearance)

    # Combined score: appearance is the stronger identity signal.
    # Geometry tells you "is this nearby?" — appearance tells you "is this the same person?"
    # Weight: 40% geometry, 60% appearance (when appearance is available)
    if app_matched >= 3:
        combined = 0.4 * geo_score + 0.6 * app_score
    else:
        combined = geo_score  # fall back to geometry only

    return combined, geo_matched


def _transform_pose_to_frame(
    pose: dict,
    crop_offset: tuple[int, int],
    crop_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> None:
    """Transform crop-local normalized coords to full-frame normalized coords in-place."""
    crop_x, crop_y = crop_offset
    crop_w, crop_h = crop_size
    full_w, full_h = frame_size

    for kp_name, kp in pose.get("dancer_pose", {}).items():
        kp["x"] = kp["x"] * (crop_w / full_w) + (crop_x / full_w)
        kp["y"] = kp["y"] * (crop_h / full_h) + (crop_y / full_h)
    for hand_key in ("left_hand", "right_hand"):
        for kp_name, kp in pose.get(hand_key, {}).items():
            kp["x"] = kp["x"] * (crop_w / full_w) + (crop_x / full_w)
            kp["y"] = kp["y"] * (crop_h / full_h) + (crop_y / full_h)
    for kp in pose.get("face", []):
        kp["x"] = kp["x"] * (crop_w / full_w) + (crop_x / full_w)
        kp["y"] = kp["y"] * (crop_h / full_h) + (crop_y / full_h)


def _bboxes_overlap(a: dict, b: dict) -> bool:
    """Check if two normalized bboxes overlap."""
    return (a["x_min"] < b["x_max"] and a["x_max"] > b["x_min"]
            and a["y_min"] < b["y_max"] and a["y_max"] > b["y_min"])


def _skeleton_centroid(pose: dict) -> tuple[float, float] | None:
    """Compute centroid of all confident body keypoints."""
    xs, ys = [], []
    for name in BODY_KEYPOINT_NAMES:
        kp = pose.get(name)
        if kp and kp.get("confidence", 0) > 0.3:
            xs.append(kp["x"])
            ys.append(kp["y"])
    if len(xs) >= 3:
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    return None


def _assign_skeletons_to_dancers(
    all_poses: list[dict],
    candidate_appearances: list[dict[str, np.ndarray]],
    dancer_states: dict[int, dict],
    dancer_bboxes_this_frame: dict[int, dict],
) -> dict[int, dict | None]:
    """Assign detected skeletons to dancers using skeleton + appearance + mutual exclusion.

    Scores every (dancer, skeleton) pair by combined identity score (geometry +
    appearance), then assigns best-first ensuring each skeleton is used at most
    once and each dancer gets at most one.
    """
    assignments: dict[int, dict | None] = {did: None for did in dancer_states}

    if not all_poses:
        return assignments

    # Build cost matrix: (dancer_id, pose_idx, score, matched_count)
    costs = []
    for did, state in dancer_states.items():
        prev_skel = state.get("prev_skeleton")
        prev_app = state.get("prev_appearance", {})
        bbox_entry = dancer_bboxes_this_frame.get(did)
        if bbox_entry is None:
            continue

        bbox = bbox_entry["bbox"]
        bbox_cx = (bbox["x_min"] + bbox["x_max"]) / 2
        bbox_cy = (bbox["y_min"] + bbox["y_max"]) / 2

        for pidx, pose in enumerate(all_poses):
            dp = pose.get("dancer_pose", {})
            if prev_skel is not None:
                score, matched = _skeleton_identity_score(
                    prev_skel, prev_app, dp, candidate_appearances[pidx],
                )
                if matched >= 3:
                    costs.append((did, pidx, score, matched))
            else:
                # No previous skeleton — use centroid distance to bbox center
                centroid = _skeleton_centroid(dp)
                if centroid:
                    d = ((centroid[0] - bbox_cx) ** 2 + (centroid[1] - bbox_cy) ** 2) ** 0.5
                    costs.append((did, pidx, d, 0))

    # Greedy assignment: best score first
    costs.sort(key=lambda x: x[2])
    used_poses: set[int] = set()
    assigned_dancers: set[int] = set()

    # Threshold for combined score. Geometry alone caps at ~0.08, appearance at ~0.3,
    # so combined (0.4*geo + 0.6*app) for a wrong person would be high.
    MAX_IDENTITY_SCORE = 0.15

    for did, pidx, score, matched in costs:
        if did in assigned_dancers or pidx in used_poses:
            continue
        if matched > 0 and score > MAX_IDENTITY_SCORE:
            continue
        assignments[did] = all_poses[pidx]
        used_poses.add(pidx)
        assigned_dancers.add(did)

    return assignments


def run_pose_estimation_multi(
    video_path: str,
    metadata: dict,
    all_dancer_bboxes: dict[int, list[dict]],
    start_ms: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[int, list[dict]]:
    """Run pose estimation for all dancers simultaneously, one frame at a time.

    Processes all dancers per-frame together so that during occlusion we can:
    - Detect overlapping bboxes and run pose on the combined region
    - Use skeleton continuity (full joint-by-joint matching) to maintain identity
    - Apply mutual exclusion — each detected skeleton belongs to at most one dancer
    - Track velocity to predict positions through occlusion

    Args:
        all_dancer_bboxes: {track_id: [{timestamp_ms, bbox, mask_iou}, ...]}
        start_ms: Video timestamp to start from.

    Returns:
        {track_id: [pose_data_dicts]} — per-dancer frame results.
    """
    model = _init_model()

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    codec = metadata.get("codec", "h264")
    BBOX_PAD = 0.25

    dancer_ids = sorted(all_dancer_bboxes.keys())

    # Build per-dancer bbox lookup by timestamp
    bbox_lookups: dict[int, dict[int, dict]] = {}
    all_timestamps: set[int] = set()
    for did, bboxes in all_dancer_bboxes.items():
        lookup = {b["timestamp_ms"]: b for b in bboxes}
        bbox_lookups[did] = lookup
        all_timestamps.update(lookup.keys())

    timestamps_sorted = sorted(all_timestamps)
    total = len(timestamps_sorted)

    if total == 0:
        return {did: [] for did in dancer_ids}

    # Build ffmpeg command
    frame_skip = SAM2_FRAME_SKIP
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    nvdec = NVDEC_CODECS.get(codec)
    if nvdec:
        cmd += ["-hwaccel", "cuda", "-c:v", nvdec]
    cmd += ["-i", video_path]
    if start_ms > 0:
        cmd += ["-ss", str(start_ms / 1000.0)]
    vf_parts = []
    if frame_skip > 1:
        vf_parts.append(f"select='not(mod(n\\,{frame_skip}))'")
        vf_parts.append("setpts=N/FRAME_RATE/TB")
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-v", "error", "pipe:1"]

    frame_bytes = width * height * 3
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Per-dancer state: previous skeleton, appearance, velocity
    dancer_states: dict[int, dict] = {
        did: {"prev_skeleton": None, "prev_appearance": {}, "velocity": (0.0, 0.0)}
        for did in dancer_ids
    }
    results: dict[int, list[dict]] = {did: [] for did in dancer_ids}
    frame_idx = 0
    total_yielded = 0
    total_rejected = 0
    occlusion_frames = 0

    logger.info(f"Multi-dancer pose estimation: {len(dancer_ids)} dancers, "
                f"{total} timestamps, start={start_ms}ms, skip={frame_skip}")

    def _update_dancer_state(did: int, pose_data: dict, frame_img: np.ndarray):
        """Update a dancer's skeleton, appearance, and velocity from a matched pose."""
        dp = pose_data.get("dancer_pose", {})
        old_centroid = _skeleton_centroid(dancer_states[did].get("prev_skeleton") or {})
        new_centroid = _skeleton_centroid(dp)
        dancer_states[did]["prev_skeleton"] = dp
        dancer_states[did]["prev_appearance"] = _extract_joint_appearance(
            dp, frame_img, width, height,
        )
        if old_centroid and new_centroid:
            dancer_states[did]["velocity"] = (
                new_centroid[0] - old_centroid[0],
                new_centroid[1] - old_centroid[1],
            )

    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break

            timestamp_ms = start_ms + int(frame_idx * frame_skip / fps * 1000)
            frame_idx += 1

            # Gather bboxes for all dancers at this timestamp
            dancer_bboxes_now: dict[int, dict] = {}
            for did in dancer_ids:
                entry = bbox_lookups[did].get(timestamp_ms)
                if entry is None:
                    ts_list = list(bbox_lookups[did].keys())
                    if ts_list:
                        nearest = min(ts_list, key=lambda t: abs(t - timestamp_ms))
                        if abs(nearest - timestamp_ms) < 50:
                            entry = bbox_lookups[did][nearest]
                if entry and entry.get("mask_iou", 0) >= 0.3:
                    bbox = entry["bbox"]
                    if bbox["x_max"] - bbox["x_min"] >= 0.01 and bbox["y_max"] - bbox["y_min"] >= 0.01:
                        dancer_bboxes_now[did] = entry

            if not dancer_bboxes_now:
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            # Detect which dancers have overlapping bboxes
            overlap_groups: list[set[int]] = []
            dids_in_frame = list(dancer_bboxes_now.keys())
            for i, did_a in enumerate(dids_in_frame):
                for did_b in dids_in_frame[i + 1:]:
                    if _bboxes_overlap(dancer_bboxes_now[did_a]["bbox"],
                                       dancer_bboxes_now[did_b]["bbox"]):
                        merged = False
                        for group in overlap_groups:
                            if did_a in group or did_b in group:
                                group.add(did_a)
                                group.add(did_b)
                                merged = True
                                break
                        if not merged:
                            overlap_groups.append({did_a, did_b})

            overlapping_dids = set()
            for group in overlap_groups:
                overlapping_dids.update(group)

            # --- Process non-overlapping dancers: crop per dancer ---
            for did in dids_in_frame:
                if did in overlapping_dids:
                    continue

                bbox = dancer_bboxes_now[did]["bbox"]
                bw = bbox["x_max"] - bbox["x_min"]
                bh = bbox["y_max"] - bbox["y_min"]
                pad_x, pad_y = bw * BBOX_PAD, bh * BBOX_PAD
                px1 = max(0, int((bbox["x_min"] - pad_x) * width))
                py1 = max(0, int((bbox["y_min"] - pad_y) * height))
                px2 = min(width, int((bbox["x_max"] + pad_x) * width))
                py2 = min(height, int((bbox["y_max"] + pad_y) * height))

                crop = frame[py1:py2, px1:px2]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                crop_h, crop_w = crop.shape[:2]
                keypoints, scores = model(crop)
                if keypoints is None or len(keypoints) == 0:
                    continue

                all_poses = _extract_all_poses(keypoints, scores, crop_w, crop_h)
                if not all_poses:
                    continue

                # Transform to full-frame coords
                for pose in all_poses:
                    _transform_pose_to_frame(pose, (px1, py1), (crop_w, crop_h), (width, height))

                prev_skel = dancer_states[did]["prev_skeleton"]
                prev_app = dancer_states[did]["prev_appearance"]
                bbox_cx = (bbox["x_min"] + bbox["x_max"]) / 2
                bbox_cy = (bbox["y_min"] + bbox["y_max"]) / 2

                best = None
                if prev_skel is not None:
                    scored = []
                    for pose in all_poses:
                        dp = pose.get("dancer_pose", {})
                        cand_app = _extract_joint_appearance(dp, frame, width, height)
                        score, matched = _skeleton_identity_score(
                            prev_skel, prev_app, dp, cand_app,
                        )
                        if matched >= 3 and score <= 0.15:
                            scored.append((score, pose))
                    if scored:
                        scored.sort(key=lambda x: x[0])
                        best = scored[0][1]
                else:
                    if len(all_poses) == 1:
                        best = all_poses[0]
                    else:
                        best_d = float("inf")
                        for pose in all_poses:
                            c = _skeleton_centroid(pose.get("dancer_pose", {}))
                            if c:
                                d = ((c[0] - bbox_cx) ** 2 + (c[1] - bbox_cy) ** 2) ** 0.5
                                if d < best_d:
                                    best_d = d
                                    best = pose

                if best is None:
                    total_rejected += 1
                    continue

                _update_dancer_state(did, best, frame)
                best["bbox"] = (
                    round(bbox["x_min"], 5), round(bbox["y_min"], 5),
                    round(bbox["x_max"], 5), round(bbox["y_max"], 5),
                )
                best["timestamp_ms"] = timestamp_ms
                results[did].append(best)
                total_yielded += 1

            # --- Process overlapping dancers: combined region + mutual exclusion ---
            for group in overlap_groups:
                occlusion_frames += 1
                group_dids = sorted(group)

                # Compute union bbox of all overlapping dancers
                union_x_min = min(dancer_bboxes_now[d]["bbox"]["x_min"] for d in group_dids)
                union_y_min = min(dancer_bboxes_now[d]["bbox"]["y_min"] for d in group_dids)
                union_x_max = max(dancer_bboxes_now[d]["bbox"]["x_max"] for d in group_dids)
                union_y_max = max(dancer_bboxes_now[d]["bbox"]["y_max"] for d in group_dids)

                uw = union_x_max - union_x_min
                uh = union_y_max - union_y_min
                pad_x, pad_y = uw * BBOX_PAD, uh * BBOX_PAD
                px1 = max(0, int((union_x_min - pad_x) * width))
                py1 = max(0, int((union_y_min - pad_y) * height))
                px2 = min(width, int((union_x_max + pad_x) * width))
                py2 = min(height, int((union_y_max + pad_y) * height))

                crop = frame[py1:py2, px1:px2]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                crop_h, crop_w = crop.shape[:2]
                keypoints, scores = model(crop)
                if keypoints is None or len(keypoints) == 0:
                    continue

                all_poses = _extract_all_poses(keypoints, scores, crop_w, crop_h)
                if not all_poses:
                    continue

                # Transform to full-frame coords
                for pose in all_poses:
                    _transform_pose_to_frame(pose, (px1, py1), (crop_w, crop_h), (width, height))

                # Extract appearance for each candidate skeleton
                cand_appearances = [
                    _extract_joint_appearance(p.get("dancer_pose", {}), frame, width, height)
                    for p in all_poses
                ]

                # Assign skeletons with mutual exclusion using identity (geometry + appearance)
                group_states = {d: dancer_states[d] for d in group_dids}
                group_bboxes = {d: dancer_bboxes_now[d] for d in group_dids}
                assigned = _assign_skeletons_to_dancers(
                    all_poses, cand_appearances, group_states, group_bboxes,
                )

                for did, pose_data in assigned.items():
                    if pose_data is None:
                        total_rejected += 1
                        continue

                    _update_dancer_state(did, pose_data, frame)
                    bbox = dancer_bboxes_now[did]["bbox"]
                    pose_data["bbox"] = (
                        round(bbox["x_min"], 5), round(bbox["y_min"], 5),
                        round(bbox["x_max"], 5), round(bbox["y_max"], 5),
                    )
                    pose_data["timestamp_ms"] = timestamp_ms
                    results[did].append(pose_data)
                    total_yielded += 1

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total)
            if is_cancelled and frame_idx % 10 == 0 and is_cancelled():
                break

    finally:
        process.stdout.close()
        process.wait()

    logger.info(f"Multi-dancer pose estimation complete: {frame_idx} frames read, "
                f"{total_yielded} poses yielded, {total_rejected} rejected, "
                f"{occlusion_frames} occlusion frames")

    return results

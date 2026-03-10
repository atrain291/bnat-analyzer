"""WHAM 3D pose estimation wrapper for Bharatanatyam analysis.

Runs WHAM in local-only mode (no SLAM) to produce per-frame:
  - 3D joint positions (24 SMPL joints in camera coordinates)
  - Foot-ground contact probabilities
  - SMPL body shape parameters

Requires: WHAM checkpoint, SMPL model files, ViTPose weights.
Falls back gracefully if WHAM is not available.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# WHAM installation path (cloned repo)
WHAM_ROOT = os.environ.get("WHAM_ROOT", "/app/third-party/WHAM")
SMPL_MODEL_PATH = os.environ.get("SMPL_MODEL_PATH", "/app/models/smpl")
WHAM_CHECKPOINT = os.environ.get("WHAM_CHECKPOINT", "/app/models/wham/wham_vit_w_3dpw.pth.tar")

_wham_available = None
_wham_model = None


def is_wham_available() -> bool:
    """Check if WHAM dependencies are installed and models exist."""
    global _wham_available
    if _wham_available is not None:
        return _wham_available

    try:
        import torch
        import smplx
        # Check for model files
        smpl_path = Path(SMPL_MODEL_PATH)
        checkpoint_path = Path(WHAM_CHECKPOINT)

        if not smpl_path.exists():
            logger.info(f"WHAM: SMPL model not found at {smpl_path}")
            _wham_available = False
            return False

        if not checkpoint_path.exists():
            logger.info(f"WHAM: Checkpoint not found at {checkpoint_path}")
            _wham_available = False
            return False

        # Check WHAM source is available
        wham_path = Path(WHAM_ROOT)
        if not wham_path.exists():
            logger.info(f"WHAM: Source not found at {wham_path}")
            _wham_available = False
            return False

        _wham_available = True
        logger.info("WHAM: Available and ready")
        return True

    except ImportError as e:
        logger.info(f"WHAM: Missing dependency: {e}")
        _wham_available = False
        return False


def _load_wham_model():
    """Load WHAM model (lazy singleton)."""
    global _wham_model
    if _wham_model is not None:
        return _wham_model

    import torch

    # Add WHAM to path
    if WHAM_ROOT not in sys.path:
        sys.path.insert(0, WHAM_ROOT)

    from lib.models import build_network
    from lib.utils.config import get_cfg
    import smplx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load WHAM config
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(WHAM_ROOT, "configs", "yamls", "demo.yaml"))
    cfg.DEVICE = str(device)

    # Build SMPL body model
    body_model = smplx.create(
        SMPL_MODEL_PATH,
        model_type="smpl",
        gender="neutral",
        batch_size=1,
    ).to(device)

    # Build WHAM network
    network = build_network(cfg, body_model)
    checkpoint = torch.load(WHAM_CHECKPOINT, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint["model"], strict=False)
    network.eval()

    _wham_model = {
        "network": network,
        "body_model": body_model,
        "cfg": cfg,
        "device": device,
    }

    logger.info("WHAM: Model loaded successfully")
    return _wham_model


def run_wham_estimation(
    video_path: str,
    metadata: dict,
    progress_callback=None,
    is_cancelled=None,
) -> list[dict] | None:
    """Run WHAM 3D pose estimation on a video.

    Args:
        video_path: Path to the input video file.
        metadata: Video metadata dict with fps, duration_ms, total_frames.
        progress_callback: Optional (current_frame, total) callback.
        is_cancelled: Optional callable returning True to abort.

    Returns:
        List of frame dicts with keys:
          - timestamp_ms: int
          - joints_3d: list of 24 [x, y, z] joint positions
          - foot_contact: dict with left_heel, left_toe, right_heel, right_toe probs
          - world_position: dict with x, y, z pelvis position (camera coords)
        Returns None if WHAM is not available or fails.
    """
    if not is_wham_available():
        logger.info("WHAM: Skipping 3D estimation (not available)")
        return None

    try:
        import torch
        model_data = _load_wham_model()
    except Exception as e:
        logger.warning(f"WHAM: Failed to load model: {e}")
        return None

    try:
        network = model_data["network"]
        body_model = model_data["body_model"]
        device = model_data["device"]
        cfg = model_data["cfg"]

        # Add WHAM to path for imports
        if WHAM_ROOT not in sys.path:
            sys.path.insert(0, WHAM_ROOT)

        from lib.data_utils.img_utils import get_single_image_crop_demo
        from lib.utils.imutils import process_image

        fps = metadata.get("fps", 30.0)
        total_frames = metadata.get("total_frames", 0)

        # Run WHAM's preprocessing: detection + 2D pose + feature extraction
        # This uses WHAM's internal ViTPose and YOLOv8
        from wham_api import WHAM_API

        # Initialize API in local-only mode
        wham_api = WHAM_API(cfg, estimate_local_only=True)

        logger.info(f"WHAM: Processing video ({total_frames} frames)")
        results, tracking_results, _ = wham_api(video_path)

        if not results:
            logger.warning("WHAM: No persons detected")
            return None

        # Take the primary tracked person (longest track)
        primary_id = max(results.keys(), key=lambda k: len(results[k].get("poses_body", [])))
        person = results[primary_id]

        poses_body = person["poses_body"]  # (T, 23, 3, 3)
        poses_root = person["poses_root_cam"]  # (T, 1, 3, 3)
        betas = person["betas"]  # (T, 10) or (10,)
        trans_cam = person["trans_cam"]  # (T, 3)

        T = len(poses_body)

        # Get 3D joint positions via SMPL forward kinematics
        with torch.no_grad():
            # Prepare SMPL inputs
            body_pose_aa = []
            for t in range(T):
                from scipy.spatial.transform import Rotation as R
                # Convert rotation matrices to axis-angle
                joints_aa = []
                for j in range(23):
                    rot = R.from_matrix(poses_body[t, j])
                    joints_aa.append(rot.as_rotvec())
                body_pose_aa.append(np.concatenate(joints_aa))

            body_pose_tensor = torch.tensor(np.array(body_pose_aa), dtype=torch.float32, device=device)

            # Root orientation
            root_aa = []
            for t in range(T):
                from scipy.spatial.transform import Rotation as R
                rot = R.from_matrix(poses_root[t, 0])
                root_aa.append(rot.as_rotvec())
            root_tensor = torch.tensor(np.array(root_aa), dtype=torch.float32, device=device)

            # Betas
            if betas.ndim == 1:
                betas_tensor = torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0).expand(T, -1)
            else:
                betas_tensor = torch.tensor(betas, dtype=torch.float32, device=device)

            transl_tensor = torch.tensor(trans_cam, dtype=torch.float32, device=device)

            # Run SMPL in batches to avoid OOM
            batch_size = 128
            all_joints = []
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                output = body_model(
                    body_pose=body_pose_tensor[start:end],
                    global_orient=root_tensor[start:end],
                    betas=betas_tensor[start:end],
                    transl=transl_tensor[start:end],
                )
                all_joints.append(output.joints[:, :24].cpu().numpy())

            joints_3d = np.concatenate(all_joints, axis=0)  # (T, 24, 3)

        # Extract foot contact if available
        contact = person.get("contact")  # (T, 4) or None

        # Build per-frame output dicts aligned to video timestamps
        # WHAM processes all frames where a person is detected;
        # tracking_results contains the frame indices
        tracking = tracking_results.get(primary_id, {})
        frame_indices = tracking.get("frame_ids", list(range(T)))

        frames_out = []
        for i in range(T):
            if is_cancelled and is_cancelled():
                logger.info("WHAM: Cancelled")
                break

            frame_idx = frame_indices[i] if i < len(frame_indices) else i
            timestamp_ms = int(round(frame_idx / fps * 1000))

            frame_dict = {
                "timestamp_ms": timestamp_ms,
                "joints_3d": joints_3d[i].tolist(),  # (24, 3) -> list of lists
                "world_position": {
                    "x": float(trans_cam[i, 0]),
                    "y": float(trans_cam[i, 1]),
                    "z": float(trans_cam[i, 2]),
                },
            }

            if contact is not None and i < len(contact):
                frame_dict["foot_contact"] = {
                    "left_heel": float(contact[i, 0]),
                    "left_toe": float(contact[i, 1]),
                    "right_heel": float(contact[i, 2]),
                    "right_toe": float(contact[i, 3]),
                }

            frames_out.append(frame_dict)

            if progress_callback and i % 50 == 0:
                progress_callback(i, T)

        logger.info(f"WHAM: Produced {len(frames_out)} frames of 3D data")
        return frames_out

    except Exception as e:
        logger.warning(f"WHAM: 3D estimation failed (non-fatal): {e}")
        return None


def merge_wham_with_rtmpose(
    rtmpose_frames: list[dict],
    wham_frames: list[dict] | None,
    tolerance_ms: int = 50,
) -> list[dict]:
    """Merge WHAM 3D data into RTMPose frame dicts by matching timestamps.

    For each RTMPose frame, find the closest WHAM frame within tolerance_ms
    and add joints_3d, world_position, foot_contact keys.

    Args:
        rtmpose_frames: Frame dicts from RTMPose with dancer_pose, etc.
        wham_frames: Frame dicts from WHAM with joints_3d, etc. May be None.
        tolerance_ms: Maximum timestamp difference for matching.

    Returns:
        The rtmpose_frames list, enriched with 3D data where available.
    """
    if not wham_frames:
        return rtmpose_frames

    # Build sorted array of WHAM timestamps for binary search
    wham_ts = np.array([f["timestamp_ms"] for f in wham_frames])

    for rtm_frame in rtmpose_frames:
        ts = rtm_frame["timestamp_ms"]

        # Binary search for closest WHAM frame
        idx = np.searchsorted(wham_ts, ts)
        best_idx = None
        best_diff = tolerance_ms + 1

        for candidate in [idx - 1, idx]:
            if 0 <= candidate < len(wham_ts):
                diff = abs(wham_ts[candidate] - ts)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = candidate

        if best_idx is not None and best_diff <= tolerance_ms:
            wf = wham_frames[best_idx]
            rtm_frame["joints_3d"] = wf.get("joints_3d")
            rtm_frame["world_position"] = wf.get("world_position")
            rtm_frame["foot_contact"] = wf.get("foot_contact")

    return rtmpose_frames

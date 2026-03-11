"""WHAM 3D pose estimation inference.

Adapted from VirtualFencer WHAM worker. Uses staged GPU loading to stay
within 12GB VRAM:
  1. Load HMR2a feature extractor -> extract features -> release
  2. Load WHAM network -> run inference -> release

Runs in --estimate_local_only mode (no DPVO/SLAM).
"""
import logging
import os
import sys

import numpy as np
import torch

logger = logging.getLogger(__name__)

WHAM_PATH = "/opt/wham"
if WHAM_PATH not in sys.path:
    sys.path.insert(0, WHAM_PATH)

# SMPL 24-joint names
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# COCO-17 joint names (matches RTMPose body keypoints)
COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _get_cfg():
    """Get WHAM config."""
    from configs.config import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(WHAM_PATH, "configs/yamls/demo.yaml"))
    cfg.TRAIN.CHECKPOINT = os.path.join(WHAM_PATH, "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar")
    return cfg


def _release_gpu():
    """Release GPU memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("GPU memory released")


def release_model():
    """Release GPU memory (public API)."""
    _release_gpu()


def _poses_to_tracking_results(poses: list[dict], width: int, height: int,
                                subject_id: int = 0) -> dict:
    """Convert pipeline pose dicts to WHAM tracking_results format.

    Args:
        poses: list of pose dicts per frame ({joint_name: {x, y, z, confidence}})
        width, height: video pixel dimensions
        subject_id: tracking ID for this subject

    Returns: tracking_results dict with per-subject data
    """
    frame_ids = []
    keypoints_list = []
    bboxes = []

    for frame_idx, pose in enumerate(poses):
        if not pose:
            continue

        # Convert to COCO-17 [x_pixel, y_pixel, confidence] array
        kp = np.zeros((17, 3), dtype=np.float32)
        valid = False
        for j, name in enumerate(COCO_NAMES):
            pt = pose.get(name, {})
            if pt and pt.get("confidence", 0) > 0.3:
                kp[j, 0] = pt["x"] * width
                kp[j, 1] = pt["y"] * height
                kp[j, 2] = pt["confidence"]
                valid = True

        if not valid:
            continue

        frame_ids.append(frame_idx)
        keypoints_list.append(kp)

        # Compute bbox as (cx, cy, scale) from keypoints
        visible = kp[:, 2] > 0.3
        if visible.any():
            x_min, x_max = kp[visible, 0].min(), kp[visible, 0].max()
            y_min, y_max = kp[visible, 1].min(), kp[visible, 1].max()
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            s = max(x_max - x_min, y_max - y_min) * 1.2 / 200.0
            bboxes.append(np.array([cx, cy, s]))
        else:
            bboxes.append(np.array([width / 2, height / 2, 1.0]))

    if len(frame_ids) < 10:
        return {}

    return {
        subject_id: {
            "frame_id": np.array(frame_ids),
            "bbox": np.array(bboxes),
            "keypoints": np.array(keypoints_list),
            "features": [],
            "flipped_bbox": [],
            "flipped_keypoints": [],
            "flipped_features": [],
        }
    }


def joints_3d_to_list(joints: np.ndarray) -> list:
    """Convert (24, 3) SMPL joint array to list of [x, y, z] lists."""
    return [[float(joints[i, 0]), float(joints[i, 1]), float(joints[i, 2])]
            for i in range(min(len(joints), 24))]


def run_wham_inference(video_path: str, poses: list[dict],
                       video_info: dict) -> dict | None:
    """Run WHAM on a sequence of frames for a single person.

    Uses staged GPU loading to stay within 12GB VRAM.

    Args:
        video_path: path to source video
        poses: list of pose dicts per frame ({joint_name: {x, y, z, confidence}})
        video_info: dict with width, height, fps

    Returns: dict with per-frame SMPL results, or None on failure
    """
    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    fps = video_info.get("fps", 30)

    # Build tracking_results from our pose data
    tracking_results = _poses_to_tracking_results(poses, width, height, subject_id=0)
    if not tracking_results:
        logger.info("Too few valid poses for WHAM inference")
        return None

    T = len(tracking_results[0]["frame_id"])

    orig_cwd = os.getcwd()
    os.chdir(WHAM_PATH)

    try:
        checkpoint_path = os.path.join(WHAM_PATH, "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar")
        if not os.path.exists(checkpoint_path):
            logger.error("WHAM checkpoint not found at %s", checkpoint_path)
            return None

        cfg = _get_cfg()

        # ---- Stage 1: Feature extraction (HMR2a ViT backbone ~2.9 GB) ----
        logger.info("Stage 1: Extracting features for %d frames...", T)
        torch.cuda.empty_cache()

        from lib.models.preproc.backbone.hmr2 import hmr2
        from lib.models.preproc.backbone.utils import process_image
        import cv2

        ckpt = os.path.join(WHAM_PATH, "checkpoints", "hmr2a.ckpt")
        hmr2_model = hmr2(ckpt).to(cfg.DEVICE).eval()

        subject_id = 0
        tr = tracking_results[subject_id]
        frame_id_set = set(tr["frame_id"].tolist())
        frame_id_to_idx = {fid: i for i, fid in enumerate(tr["frame_id"])}

        features = []
        init_global_orient = None
        init_body_pose = None
        init_betas = None

        cap = cv2.VideoCapture(video_path)
        video_frame_idx = 0
        processed = 0

        while True:
            ret, img = cap.read()
            if not ret:
                break

            if video_frame_idx in frame_id_set:
                idx = frame_id_to_idx[video_frame_idx]
                bbox = tr["bbox"][idx]
                cx, cy, scale = bbox

                norm_img, _ = process_image(img[..., ::-1], [cx, cy], scale, 256, 256)
                norm_tensor = torch.from_numpy(norm_img).unsqueeze(0).to(cfg.DEVICE)

                with torch.no_grad():
                    feature = hmr2_model(norm_tensor, encode=True)
                features.append(feature.cpu())
                del norm_tensor, feature

                # Get initial pose from first frame
                if init_global_orient is None:
                    norm_tensor2 = torch.from_numpy(norm_img).unsqueeze(0).to(cfg.DEVICE)
                    with torch.no_grad():
                        go, bp, bt, _ = hmr2_model(norm_tensor2, encode=False)
                    init_global_orient = go.cpu()
                    init_body_pose = bp.cpu()
                    init_betas = bt.cpu()
                    del norm_tensor2, go, bp, bt

                processed += 1
                if processed % 50 == 0:
                    torch.cuda.empty_cache()
                    logger.info("Feature extraction: %d/%d frames", processed, T)

            video_frame_idx += 1

        cap.release()

        tracking_results[subject_id]["features"] = torch.cat(features) if features else torch.zeros(0)
        tracking_results[subject_id]["init_global_orient"] = init_global_orient
        tracking_results[subject_id]["init_body_pose"] = init_body_pose
        tracking_results[subject_id]["init_betas"] = init_betas

        del hmr2_model, features
        _release_gpu()
        logger.info("Stage 1 complete: %d features extracted, extractor released", processed)

        # ---- Stage 2: WHAM network inference (~0.2 GB) ----
        logger.info("Stage 2: Running WHAM network inference...")
        from lib.models import build_network, build_body_model
        from lib.data.datasets import CustomDataset
        from lib.utils.transforms import matrix_to_axis_angle

        smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
        smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
        network = build_network(cfg, smpl)
        network.eval()

        # Build dummy SLAM results (no DPVO — estimate_local_only)
        total_frames = len(poses)
        slam_results = np.zeros((total_frames, 7), dtype=np.float32)
        slam_results[:, 3] = 1.0  # Unit quaternion

        dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)

        with torch.no_grad():
            batch = dataset.load_data(0)
            _id, x, inits, feats, mask, init_root, cam_angvel, frame_id, kwargs = batch
            pred = network(x, inits, feats, mask=mask, init_root=init_root,
                          cam_angvel=cam_angvel, return_y_up=True, **kwargs)

        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_joints = network.output.joints.cpu() if hasattr(network, 'output') and hasattr(network.output, 'joints') else None
        foot_contact = pred.get('contact')

        del network, smpl, dataset
        _release_gpu()
        logger.info("Stage 2 complete, network released")

        # ---- Build results dict ----
        frame_ids = tracking_results[0]["frame_id"]
        n_output = pred_body_pose.shape[0]

        results = {
            "joints_3d": [],
            "foot_contacts": [],
            "world_positions": [],
            "frame_count": n_output,
            "frame_ids": frame_ids.tolist() if hasattr(frame_ids, 'tolist') else frame_ids,
        }

        for t in range(n_output):
            if pred_joints is not None:
                if pred_joints.dim() == 3:
                    j3d = pred_joints[t].numpy()
                elif pred_joints.dim() == 4:
                    j3d = pred_joints[0, t].numpy()
                else:
                    j3d = pred_joints[t].numpy()
                results["joints_3d"].append(joints_3d_to_list(j3d))
                # World position = pelvis (joint 0)
                results["world_positions"].append({
                    "x": float(j3d[0, 0]),
                    "y": float(j3d[0, 1]),
                    "z": float(j3d[0, 2]),
                })
            else:
                results["joints_3d"].append(None)
                results["world_positions"].append(None)

            if foot_contact is not None:
                fc = foot_contact[0, t].cpu().tolist()
                results["foot_contacts"].append({
                    "left_heel": fc[0] if len(fc) > 0 else 0.0,
                    "left_toe": fc[1] if len(fc) > 1 else 0.0,
                    "right_heel": fc[2] if len(fc) > 2 else 0.0,
                    "right_toe": fc[3] if len(fc) > 3 else 0.0,
                })
            else:
                results["foot_contacts"].append(None)

        logger.info("WHAM inference complete: %d frames processed", n_output)
        return results

    except Exception:
        logger.exception("WHAM inference failed")
        return None
    finally:
        os.chdir(orig_cwd)

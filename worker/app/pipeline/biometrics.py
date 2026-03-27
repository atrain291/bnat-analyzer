"""Biometric body-proportion signatures for person re-identification.

Computes pose-invariant, scale-invariant limb ratios from RTMPose keypoints.
These ratios are unique per individual and stable across frames/poses.
"""
import logging
from dataclasses import dataclass, fields

import numpy as np

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.5
MIN_RATIOS_REQUIRED = 3

# Typical ranges for each ratio (used to normalize differences in similarity)
_RATIO_RANGES = {
    "shoulder_hip_ratio": 0.6,
    "torso_leg_ratio": 0.5,
    "upper_lower_arm_ratio": 0.4,
    "thigh_shin_ratio": 0.4,
    "head_shoulder_ratio": 0.5,
    "arm_body_ratio": 0.5,
}


@dataclass
class BiometricSignature:
    shoulder_hip_ratio: float | None = None
    torso_leg_ratio: float | None = None
    upper_lower_arm_ratio: float | None = None
    thigh_shin_ratio: float | None = None
    head_shoulder_ratio: float | None = None
    arm_body_ratio: float | None = None
    available_count: int = 0


def _kp_dist(pose: dict, name_a: str, name_b: str) -> float | None:
    a = pose.get(name_a)
    b = pose.get(name_b)
    if not a or not b:
        return None
    if a.get("confidence", 0) < MIN_CONFIDENCE or b.get("confidence", 0) < MIN_CONFIDENCE:
        return None
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    d = (dx * dx + dy * dy) ** 0.5
    return d if d > 1e-6 else None


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den < 1e-6:
        return None
    return num / den


def _avg_optional(a: float | None, b: float | None) -> float | None:
    if a and b:
        return (a + b) / 2
    return a or b


def extract_biometric_signature(pose: dict) -> BiometricSignature | None:
    """Extract body-proportion ratios from a pose dict (normalized 0-1 keypoints).

    Returns None if fewer than MIN_RATIOS_REQUIRED ratios are computable.
    """
    shoulder_w = _kp_dist(pose, "left_shoulder", "right_shoulder")
    hip_w = _kp_dist(pose, "left_hip", "right_hip")

    # Torso: midpoint of shoulders to midpoint of hips
    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    lh = pose.get("left_hip")
    rh = pose.get("right_hip")
    torso_len = None
    if (ls and rs and lh and rh
            and ls.get("confidence", 0) >= MIN_CONFIDENCE
            and rs.get("confidence", 0) >= MIN_CONFIDENCE
            and lh.get("confidence", 0) >= MIN_CONFIDENCE
            and rh.get("confidence", 0) >= MIN_CONFIDENCE):
        smx = (ls["x"] + rs["x"]) / 2
        smy = (ls["y"] + rs["y"]) / 2
        hmx = (lh["x"] + rh["x"]) / 2
        hmy = (lh["y"] + rh["y"]) / 2
        torso_len = ((smx - hmx) ** 2 + (smy - hmy) ** 2) ** 0.5
        if torso_len < 1e-6:
            torso_len = None

    # Leg lengths (average left + right)
    l_thigh = _kp_dist(pose, "left_hip", "left_knee")
    r_thigh = _kp_dist(pose, "right_hip", "right_knee")
    l_shin = _kp_dist(pose, "left_knee", "left_ankle")
    r_shin = _kp_dist(pose, "right_knee", "right_ankle")
    avg_thigh = None
    if l_thigh and r_thigh:
        avg_thigh = (l_thigh + r_thigh) / 2
    elif l_thigh:
        avg_thigh = l_thigh
    elif r_thigh:
        avg_thigh = r_thigh
    avg_shin = None
    if l_shin and r_shin:
        avg_shin = (l_shin + r_shin) / 2
    elif l_shin:
        avg_shin = l_shin
    elif r_shin:
        avg_shin = r_shin
    avg_leg = None
    if avg_thigh and avg_shin:
        avg_leg = avg_thigh + avg_shin

    # Upper/lower arm (average left + right)
    l_upper = _kp_dist(pose, "left_shoulder", "left_elbow")
    r_upper = _kp_dist(pose, "right_shoulder", "right_elbow")
    l_lower = _kp_dist(pose, "left_elbow", "left_wrist")
    r_lower = _kp_dist(pose, "right_elbow", "right_wrist")
    avg_upper_arm = None
    if l_upper and r_upper:
        avg_upper_arm = (l_upper + r_upper) / 2
    elif l_upper:
        avg_upper_arm = l_upper
    elif r_upper:
        avg_upper_arm = r_upper
    avg_lower_arm = None
    if l_lower and r_lower:
        avg_lower_arm = (l_lower + r_lower) / 2
    elif l_lower:
        avg_lower_arm = l_lower
    elif r_lower:
        avg_lower_arm = r_lower

    # Head width: ear-to-ear
    head_w = _kp_dist(pose, "left_ear", "right_ear")

    # Arm length: shoulder to wrist (average)
    l_arm = None
    if l_upper and l_lower:
        l_arm = l_upper + l_lower
    r_arm = None
    if r_upper and r_lower:
        r_arm = r_upper + r_lower
    avg_arm_len = None
    if l_arm and r_arm:
        avg_arm_len = (l_arm + r_arm) / 2
    elif l_arm:
        avg_arm_len = l_arm
    elif r_arm:
        avg_arm_len = r_arm

    sig = BiometricSignature()
    count = 0

    sig.shoulder_hip_ratio = _safe_ratio(shoulder_w, hip_w)
    if sig.shoulder_hip_ratio is not None:
        count += 1

    sig.torso_leg_ratio = _safe_ratio(torso_len, avg_leg)
    if sig.torso_leg_ratio is not None:
        count += 1

    sig.upper_lower_arm_ratio = _safe_ratio(avg_upper_arm, avg_lower_arm)
    if sig.upper_lower_arm_ratio is not None:
        count += 1

    sig.thigh_shin_ratio = _safe_ratio(avg_thigh, avg_shin)
    if sig.thigh_shin_ratio is not None:
        count += 1

    sig.head_shoulder_ratio = _safe_ratio(head_w, shoulder_w)
    if sig.head_shoulder_ratio is not None:
        count += 1

    sig.arm_body_ratio = _safe_ratio(avg_arm_len, torso_len)
    if sig.arm_body_ratio is not None:
        count += 1

    sig.available_count = count

    if count < MIN_RATIOS_REQUIRED:
        return None
    return sig


def _joint_dist_3d(joints: list, idx_a: int, idx_b: int) -> float | None:
    if idx_a >= len(joints) or idx_b >= len(joints):
        return None
    a, b = joints[idx_a], joints[idx_b]
    if not a or not b or len(a) < 3 or len(b) < 3:
        return None
    d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
    return d if d > 1e-6 else None


def _midpoint_3d(joints: list, idx_a: int, idx_b: int) -> list | None:
    if idx_a >= len(joints) or idx_b >= len(joints):
        return None
    a, b = joints[idx_a], joints[idx_b]
    if not a or not b or len(a) < 3 or len(b) < 3:
        return None
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]


def extract_biometric_signature_3d(joints_3d: list[list[float]]) -> BiometricSignature | None:
    """Extract body-proportion ratios from SMPL 24-joint 3D positions.

    Uses true 3D Euclidean distances — pose-invariant unlike 2D projections.
    SMPL joints: 0=pelvis, 1=L_hip, 2=R_hip, 3=spine1, 4=L_knee, 5=R_knee,
    6=spine2, 7=L_ankle, 8=R_ankle, 9=spine3, 12=neck, 15=head,
    16=L_shoulder, 17=R_shoulder, 18=L_elbow, 19=R_elbow, 20=L_wrist, 21=R_wrist.

    Returns None if fewer than MIN_RATIOS_REQUIRED ratios are computable.
    """
    if not joints_3d or len(joints_3d) < 22:
        return None

    shoulder_w = _joint_dist_3d(joints_3d, 16, 17)
    hip_w = _joint_dist_3d(joints_3d, 1, 2)

    # Torso: midpoint of shoulders to midpoint of hips
    shoulder_mid = _midpoint_3d(joints_3d, 16, 17)
    hip_mid = _midpoint_3d(joints_3d, 1, 2)
    torso_len = None
    if shoulder_mid and hip_mid:
        d = ((shoulder_mid[0] - hip_mid[0]) ** 2 +
             (shoulder_mid[1] - hip_mid[1]) ** 2 +
             (shoulder_mid[2] - hip_mid[2]) ** 2) ** 0.5
        torso_len = d if d > 1e-6 else None

    # Legs
    l_thigh = _joint_dist_3d(joints_3d, 1, 4)
    r_thigh = _joint_dist_3d(joints_3d, 2, 5)
    l_shin = _joint_dist_3d(joints_3d, 4, 7)
    r_shin = _joint_dist_3d(joints_3d, 5, 8)
    avg_thigh = _avg_optional(l_thigh, r_thigh)
    avg_shin = _avg_optional(l_shin, r_shin)
    avg_leg = (avg_thigh + avg_shin) if (avg_thigh and avg_shin) else None

    # Arms
    l_upper = _joint_dist_3d(joints_3d, 16, 18)
    r_upper = _joint_dist_3d(joints_3d, 17, 19)
    l_lower = _joint_dist_3d(joints_3d, 18, 20)
    r_lower = _joint_dist_3d(joints_3d, 19, 21)
    avg_upper_arm = _avg_optional(l_upper, r_upper)
    avg_lower_arm = _avg_optional(l_lower, r_lower)

    # Head: head-to-neck distance (SMPL has no ear joints)
    head_neck = _joint_dist_3d(joints_3d, 15, 12)

    # Full arm length
    l_arm = (l_upper + l_lower) if (l_upper and l_lower) else None
    r_arm = (r_upper + r_lower) if (r_upper and r_lower) else None
    avg_arm_len = _avg_optional(l_arm, r_arm)

    sig = BiometricSignature()
    count = 0

    sig.shoulder_hip_ratio = _safe_ratio(shoulder_w, hip_w)
    if sig.shoulder_hip_ratio is not None:
        count += 1
    sig.torso_leg_ratio = _safe_ratio(torso_len, avg_leg)
    if sig.torso_leg_ratio is not None:
        count += 1
    sig.upper_lower_arm_ratio = _safe_ratio(avg_upper_arm, avg_lower_arm)
    if sig.upper_lower_arm_ratio is not None:
        count += 1
    sig.thigh_shin_ratio = _safe_ratio(avg_thigh, avg_shin)
    if sig.thigh_shin_ratio is not None:
        count += 1
    sig.head_shoulder_ratio = _safe_ratio(head_neck, shoulder_w)
    if sig.head_shoulder_ratio is not None:
        count += 1
    sig.arm_body_ratio = _safe_ratio(avg_arm_len, torso_len)
    if sig.arm_body_ratio is not None:
        count += 1

    sig.available_count = count
    if count < MIN_RATIOS_REQUIRED:
        return None
    return sig


def merge_signatures(existing: BiometricSignature, new: BiometricSignature,
                     alpha: float = 0.1) -> BiometricSignature:
    """EMA update of biometric signature."""
    result = BiometricSignature()
    count = 0
    for f in fields(BiometricSignature):
        if f.name == "available_count":
            continue
        old_val = getattr(existing, f.name)
        new_val = getattr(new, f.name)
        if new_val is not None and old_val is not None:
            merged = old_val * (1 - alpha) + new_val * alpha
            setattr(result, f.name, merged)
            count += 1
        elif new_val is not None:
            setattr(result, f.name, new_val)
            count += 1
        elif old_val is not None:
            setattr(result, f.name, old_val)
            count += 1
    result.available_count = count
    return result


def signature_similarity(a: BiometricSignature | None, b: BiometricSignature | None) -> float:
    """Compare two biometric signatures. Returns 0.0-1.0 (1.0 = identical).

    Only compares mutually-available ratios. Returns 0.5 if insufficient data.
    """
    if a is None or b is None:
        return 0.5
    diffs = []
    for f in fields(BiometricSignature):
        if f.name == "available_count":
            continue
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        if va is not None and vb is not None:
            range_val = _RATIO_RANGES.get(f.name, 0.5)
            diff = abs(va - vb) / range_val
            diffs.append(min(diff, 1.0))
    if len(diffs) < 2:
        return 0.5
    avg_diff = sum(diffs) / len(diffs)
    return max(0.0, 1.0 - avg_diff)

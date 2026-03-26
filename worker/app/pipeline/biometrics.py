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

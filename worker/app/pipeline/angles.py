"""Compute joint angles and pose statistics from COCO-WholeBody keypoints.

Provides per-frame angle computation and aggregate statistics
for feeding into the LLM coaching prompt. Supports 23 body+feet keypoints
from RTMPose WholeBody (133-point model).
"""

import math


def _angle_between(p1: dict, p2: dict, p3: dict) -> float | None:
    """Compute angle at p2 formed by p1-p2-p3 in degrees.

    Returns None if any point has low confidence.
    """
    min_conf = 0.3
    if p1.get("confidence", 0) < min_conf or p2.get("confidence", 0) < min_conf or p3.get("confidence", 0) < min_conf:
        return None

    v1 = (p1["x"] - p2["x"], p1["y"] - p2["y"])
    v2 = (p3["x"] - p2["x"], p3["y"] - p2["y"])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag1 < 1e-6 or mag2 < 1e-6:
        return None

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def compute_frame_angles(pose: dict) -> dict:
    """Compute Bharatanatyam-relevant angles from a single frame's keypoints.

    Returns a dict with angle measurements (degrees). Missing keypoints
    result in None values for the affected angles.
    """
    if not pose:
        return {}

    angles = {}

    # Aramandi: average knee angle (hip-knee-ankle)
    left_knee = _angle_between(
        pose.get("left_hip", {}),
        pose.get("left_knee", {}),
        pose.get("left_ankle", {}),
    )
    right_knee = _angle_between(
        pose.get("right_hip", {}),
        pose.get("right_knee", {}),
        pose.get("right_ankle", {}),
    )
    angles["left_knee_angle"] = left_knee
    angles["right_knee_angle"] = right_knee

    if left_knee is not None and right_knee is not None:
        angles["avg_knee_angle"] = (left_knee + right_knee) / 2
    elif left_knee is not None:
        angles["avg_knee_angle"] = left_knee
    elif right_knee is not None:
        angles["avg_knee_angle"] = right_knee

    # Torso uprightness: angle between vertical and shoulder-hip midline
    ls = pose.get("left_shoulder", {})
    rs = pose.get("right_shoulder", {})
    lh = pose.get("left_hip", {})
    rh = pose.get("right_hip", {})

    if all(p.get("confidence", 0) > 0.3 for p in [ls, rs, lh, rh]):
        mid_shoulder = ((ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2)
        mid_hip = ((lh["x"] + rh["x"]) / 2, (lh["y"] + rh["y"]) / 2)

        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_shoulder[1] - mid_hip[1]
        # Angle from vertical (dy is negative when upright in image coords)
        if abs(dy) > 1e-6:
            torso_angle = abs(math.degrees(math.atan2(dx, -dy)))
            angles["torso_angle"] = torso_angle

    # Arm extension: shoulder-elbow-wrist angle (180 = fully extended)
    left_arm = _angle_between(
        pose.get("left_shoulder", {}),
        pose.get("left_elbow", {}),
        pose.get("left_wrist", {}),
    )
    right_arm = _angle_between(
        pose.get("right_shoulder", {}),
        pose.get("right_elbow", {}),
        pose.get("right_wrist", {}),
    )
    angles["arm_extension_left"] = left_arm
    angles["arm_extension_right"] = right_arm

    # Hip symmetry: difference in hip Y positions (0 = level)
    if lh.get("confidence", 0) > 0.3 and rh.get("confidence", 0) > 0.3:
        hip_diff = abs(lh["y"] - rh["y"])
        # Normalize by torso length for scale-invariance
        if "torso_angle" in angles:
            torso_len = math.sqrt(
                (mid_shoulder[0] - mid_hip[0]) ** 2 + (mid_shoulder[1] - mid_hip[1]) ** 2
            )
            if torso_len > 1e-6:
                angles["hip_symmetry"] = hip_diff / torso_len
            else:
                angles["hip_symmetry"] = hip_diff
        else:
            angles["hip_symmetry"] = hip_diff

    # --- Foot angles (require WholeBody 133-point keypoints) ---

    # Foot turnout: angle formed by heel-big_toe line relative to forward axis
    # Important for aramandi — feet should turn outward ~45-60 degrees
    for side in ("left", "right"):
        heel = pose.get(f"{side}_heel", {})
        big_toe = pose.get(f"{side}_big_toe", {})
        ankle = pose.get(f"{side}_ankle", {})

        if (heel.get("confidence", 0) > 0.3
                and big_toe.get("confidence", 0) > 0.3):
            # Foot direction vector (heel -> big toe)
            foot_dx = big_toe["x"] - heel["x"]
            foot_dy = big_toe["y"] - heel["y"]
            # Angle from vertical (0 = pointing up, 90 = pointing sideways)
            if abs(foot_dx) > 1e-6 or abs(foot_dy) > 1e-6:
                foot_angle = math.degrees(math.atan2(abs(foot_dx), -foot_dy))
                angles[f"{side}_foot_turnout"] = foot_angle

        # Foot flatness: vertical distance between heel and big_toe
        # Small difference = flat foot, large difference = on toes or heels
        if (heel.get("confidence", 0) > 0.3
                and big_toe.get("confidence", 0) > 0.3):
            angles[f"{side}_foot_flatness"] = abs(heel["y"] - big_toe["y"])

        # Ankle-heel-toe angle: measures foot flexion
        # ~180 = flat foot, <180 = foot pointed, >180 = foot flexed
        if ankle.get("confidence", 0) > 0.3:
            foot_angle = _angle_between(ankle, heel, big_toe)
            if foot_angle is not None:
                angles[f"{side}_foot_angle"] = foot_angle

    return angles


def summarize_pose_statistics(frames_data: list[dict]) -> dict:
    """Aggregate angle statistics across all frames for LLM context.

    Args:
        frames_data: List of frame dicts with "dancer_pose" keys.

    Returns:
        Dict of aggregate statistics suitable for the LLM prompt.
    """
    knee_angles = []
    torso_angles = []
    arm_left = []
    arm_right = []
    hip_sym = []
    foot_turnout_left = []
    foot_turnout_right = []
    foot_flatness_left = []
    foot_flatness_right = []

    for frame in frames_data:
        pose = frame.get("dancer_pose", {})
        if not pose:
            continue

        angles = compute_frame_angles(pose)

        if "avg_knee_angle" in angles:
            knee_angles.append(angles["avg_knee_angle"])
        if "torso_angle" in angles:
            torso_angles.append(angles["torso_angle"])
        if angles.get("arm_extension_left") is not None:
            arm_left.append(angles["arm_extension_left"])
        if angles.get("arm_extension_right") is not None:
            arm_right.append(angles["arm_extension_right"])
        if "hip_symmetry" in angles:
            hip_sym.append(angles["hip_symmetry"])
        if "left_foot_turnout" in angles:
            foot_turnout_left.append(angles["left_foot_turnout"])
        if "right_foot_turnout" in angles:
            foot_turnout_right.append(angles["right_foot_turnout"])
        if "left_foot_flatness" in angles:
            foot_flatness_left.append(angles["left_foot_flatness"])
        if "right_foot_flatness" in angles:
            foot_flatness_right.append(angles["right_foot_flatness"])

    summary = {}

    if knee_angles:
        summary["avg_knee_angle"] = sum(knee_angles) / len(knee_angles)
        summary["min_knee_angle"] = min(knee_angles)
        summary["max_knee_angle"] = max(knee_angles)
        if len(knee_angles) > 1:
            mean = summary["avg_knee_angle"]
            variance = sum((a - mean) ** 2 for a in knee_angles) / len(knee_angles)
            summary["knee_angle_std"] = math.sqrt(variance)

    if torso_angles:
        summary["avg_torso_angle"] = sum(torso_angles) / len(torso_angles)

    if arm_left:
        summary["avg_arm_extension_left"] = sum(arm_left) / len(arm_left)

    if arm_right:
        summary["avg_arm_extension_right"] = sum(arm_right) / len(arm_right)

    if hip_sym:
        summary["hip_symmetry_avg"] = sum(hip_sym) / len(hip_sym)

    # Foot turnout (degrees from vertical, higher = more turned out)
    if foot_turnout_left:
        summary["avg_foot_turnout_left"] = sum(foot_turnout_left) / len(foot_turnout_left)
    if foot_turnout_right:
        summary["avg_foot_turnout_right"] = sum(foot_turnout_right) / len(foot_turnout_right)
    if foot_turnout_left and foot_turnout_right:
        all_turnout = foot_turnout_left + foot_turnout_right
        summary["avg_foot_turnout"] = sum(all_turnout) / len(all_turnout)

    # Foot flatness (lower = flatter, which is better for flat strikes)
    if foot_flatness_left and foot_flatness_right:
        all_flatness = foot_flatness_left + foot_flatness_right
        summary["avg_foot_flatness"] = sum(all_flatness) / len(all_flatness)

    # Composite balance score (0-1)
    balance_components = []
    if torso_angles:
        # Score: 1.0 for <2 deg deviation, 0 for >15 deg
        avg_torso = summary["avg_torso_angle"]
        balance_components.append(max(0.0, min(1.0, 1.0 - (avg_torso - 2) / 13)))
    if hip_sym:
        # Score: 1.0 for <0.02 deviation, 0 for >0.15
        avg_hip = summary["hip_symmetry_avg"]
        balance_components.append(max(0.0, min(1.0, 1.0 - (avg_hip - 0.02) / 0.13)))
    if balance_components:
        summary["balance_score"] = sum(balance_components) / len(balance_components)

    return summary

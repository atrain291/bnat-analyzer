"""Compute joint angles and pose statistics from COCO-WholeBody keypoints.

Provides per-frame angle computation and aggregate statistics
for feeding into the LLM coaching prompt. Uses body+feet (23 keypoints),
hands (42 keypoints), and face (68 landmarks) from RTMPose WholeBody (133-point model).
"""

import math

import numpy as np


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


def _head_tilt_angles(face: list) -> dict:
    """Compute head tilt angles from 68-point face landmarks.

    Returns dict with:
      - head_lateral_tilt: degrees from vertical (0 = upright, positive = tilted right)
      - head_forward_tilt: ratio indicating forward/back pitch (lower = more forward)
    """
    if not face or len(face) < 31:
        return {}

    min_conf = 0.3
    chin = face[8]       # Bottom of chin
    nose_top = face[27]  # Top of nose bridge
    nose_tip = face[30]  # Tip of nose

    if any(p.get("confidence", 0) < min_conf for p in [chin, nose_top, nose_tip]):
        return {}

    result = {}

    # Lateral tilt (roll): angle of chin-to-nose_top line from vertical
    dx = nose_top["x"] - chin["x"]
    dy = nose_top["y"] - chin["y"]
    if abs(dy) > 1e-6 or abs(dx) > 1e-6:
        result["head_lateral_tilt"] = math.degrees(math.atan2(dx, -dy))

    # Forward/back tilt (pitch): ratio of nose_tip-to-chin vs nose_top-to-chin
    dist_tip_chin = math.sqrt(
        (nose_tip["x"] - chin["x"]) ** 2 + (nose_tip["y"] - chin["y"]) ** 2
    )
    dist_top_chin = math.sqrt(
        (nose_top["x"] - chin["x"]) ** 2 + (nose_top["y"] - chin["y"]) ** 2
    )
    if dist_top_chin > 1e-6:
        result["head_forward_tilt"] = dist_tip_chin / dist_top_chin

    return result


def _compute_finger_extension(hand: dict) -> dict:
    """Compute finger extension angles for a single hand.

    Returns dict with per-finger PIP angles and average extension.
    ~180° = fully extended, ~90° = curled.
    """
    if not hand:
        return {}

    results = {}
    pip_angles = []

    for finger in ("index", "middle", "ring", "pinky"):
        angle = _angle_between(
            hand.get(f"{finger}_mcp", {}),
            hand.get(f"{finger}_pip", {}),
            hand.get(f"{finger}_dip", {}),
        )
        if angle is not None:
            results[f"{finger}_pip_angle"] = angle
            pip_angles.append(angle)

    if pip_angles:
        results["avg_finger_extension"] = sum(pip_angles) / len(pip_angles)

    thumb_angle = _angle_between(
        hand.get("thumb_mcp", {}),
        hand.get("thumb_ip", {}),
        hand.get("thumb_tip", {}),
    )
    if thumb_angle is not None:
        results["thumb_extension"] = thumb_angle

    return results


def _angle_3d(p1, p2, p3):
    """Angle at p2 in the triangle p1-p2-p3, in degrees. Inputs are (3,) numpy arrays."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _compute_3d_angles(joints_3d) -> dict:
    """Compute 3D angles from SMPL 24-joint positions.

    Args:
        joints_3d: numpy array of shape (24, 3) with SMPL joint positions.

    Returns:
        Dict of 3D angle measurements keyed with ``_3d`` suffix where
        appropriate so they can coexist with 2D values.
    """
    j = joints_3d  # shorthand
    angles = {}

    # --- Knee angles ---
    left_knee = _angle_3d(j[1], j[4], j[7])    # left_hip → left_knee → left_ankle
    right_knee = _angle_3d(j[2], j[5], j[8])   # right_hip → right_knee → right_ankle
    angles["left_knee_angle_3d"] = left_knee
    angles["right_knee_angle_3d"] = right_knee
    angles["knee_angle_3d"] = (left_knee + right_knee) / 2

    # --- Torso angle: pelvis→spine3 vs world up [0, 1, 0] ---
    spine_vec = j[9] - j[0]  # pelvis → spine3
    up = np.array([0.0, 1.0, 0.0])
    cos_torso = np.dot(spine_vec, up) / (np.linalg.norm(spine_vec) + 1e-8)
    angles["torso_angle_3d"] = float(np.degrees(np.arccos(np.clip(cos_torso, -1.0, 1.0))))

    # --- Arm extension ---
    angles["arm_extension_left_3d"] = _angle_3d(j[16], j[18], j[20])   # shoulder→elbow→wrist
    angles["arm_extension_right_3d"] = _angle_3d(j[17], j[19], j[21])

    # --- Hip abduction ---
    angles["hip_abduction_left"] = _angle_3d(j[3], j[1], j[4])    # spine1→left_hip→left_knee
    angles["hip_abduction_right"] = _angle_3d(j[3], j[2], j[5])   # spine1→right_hip→right_knee

    # --- Hip symmetry (Y-height difference) ---
    angles["hip_symmetry_3d"] = float(abs(j[1][1] - j[2][1]))

    # --- Torso twist: angle between shoulder line and hip line projected onto XZ plane ---
    shoulder_line = j[17] - j[16]  # right_shoulder - left_shoulder
    hip_line = j[2] - j[1]         # right_hip - left_hip
    # Project onto XZ plane (keep X and Z, drop Y)
    s_xz = np.array([shoulder_line[0], shoulder_line[2]])
    h_xz = np.array([hip_line[0], hip_line[2]])
    cos_twist = np.dot(s_xz, h_xz) / (np.linalg.norm(s_xz) * np.linalg.norm(h_xz) + 1e-8)
    angles["torso_twist"] = float(np.degrees(np.arccos(np.clip(cos_twist, -1.0, 1.0))))

    return angles


def compute_frame_angles(
    pose: dict,
    face: list | None = None,
    left_hand: dict | None = None,
    right_hand: dict | None = None,
    joints_3d=None,
) -> dict:
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

    # --- Shoulder elevation (ear-to-shoulder distance, normalized by torso) ---
    # Higher value = more relaxed (good), lower = raised/tense shoulders (bad)
    le = pose.get("left_ear", {})
    re = pose.get("right_ear", {})

    for side, ear, shoulder, hip in [
        ("left", le, ls, lh),
        ("right", re, rs, rh),
    ]:
        if (ear.get("confidence", 0) > 0.3
                and shoulder.get("confidence", 0) > 0.3):
            ear_to_shoulder = ear["y"] - shoulder["y"]
            if hip.get("confidence", 0) > 0.3:
                torso_side_len = math.sqrt(
                    (shoulder["x"] - hip["x"]) ** 2 + (shoulder["y"] - hip["y"]) ** 2
                )
                if torso_side_len > 1e-6:
                    angles[f"{side}_shoulder_elevation"] = ear_to_shoulder / torso_side_len
                else:
                    angles[f"{side}_shoulder_elevation"] = ear_to_shoulder
            else:
                angles[f"{side}_shoulder_elevation"] = ear_to_shoulder

    l_elev = angles.get("left_shoulder_elevation")
    r_elev = angles.get("right_shoulder_elevation")
    if l_elev is not None and r_elev is not None:
        angles["shoulder_elevation_avg"] = (l_elev + r_elev) / 2
    elif l_elev is not None:
        angles["shoulder_elevation_avg"] = l_elev
    elif r_elev is not None:
        angles["shoulder_elevation_avg"] = r_elev

    # --- Neck lateral tilt (attami): shoulder-midpoint to nose angle from vertical ---
    # 0° = head centered, positive = tilted right, negative = tilted left
    nose = pose.get("nose", {})
    if (nose.get("confidence", 0) > 0.3
            and ls.get("confidence", 0) > 0.3
            and rs.get("confidence", 0) > 0.3):
        mid_sx = (ls["x"] + rs["x"]) / 2
        mid_sy = (ls["y"] + rs["y"]) / 2
        neck_dx = nose["x"] - mid_sx
        neck_dy = nose["y"] - mid_sy
        if abs(neck_dy) > 1e-6:
            tilt = math.degrees(math.atan2(neck_dx, -neck_dy))
            angles["neck_lateral_tilt"] = tilt
            angles["neck_lateral_tilt_abs"] = abs(tilt)

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

    # --- Head tilt (shirobheda) from face landmarks ---
    if face:
        head_angles = _head_tilt_angles(face)
        angles.update(head_angles)

    # --- Wrist flexion (elbow → hand_wrist → middle_mcp) ---
    # ~180° = straight wrist, <180° = flexed
    for side, hand in (("left", left_hand), ("right", right_hand)):
        if not hand:
            continue
        elbow = pose.get(f"{side}_elbow", {})
        hand_wrist = hand.get("wrist", {})
        middle_mcp = hand.get("middle_mcp", {})
        wrist_flexion = _angle_between(elbow, hand_wrist, middle_mcp)
        if wrist_flexion is not None:
            angles[f"{side}_wrist_flexion"] = wrist_flexion

    # --- Finger extension (PIP angles per hand) ---
    for side, hand in (("left", left_hand), ("right", right_hand)):
        finger_data = _compute_finger_extension(hand)
        if "avg_finger_extension" in finger_data:
            angles[f"{side}_finger_extension"] = finger_data["avg_finger_extension"]
        if "thumb_extension" in finger_data:
            angles[f"{side}_thumb_extension"] = finger_data["thumb_extension"]

    # --- 3D angles from WHAM (SMPL 24 joints) ---
    if joints_3d is not None:
        try:
            j3d = np.asarray(joints_3d, dtype=np.float64)
            if j3d.shape == (24, 3):
                angles_3d = _compute_3d_angles(j3d)
                angles.update(angles_3d)
        except (ValueError, TypeError):
            pass  # Silently skip if joints_3d is malformed

    return angles


def summarize_pose_statistics(frames_data: list[dict]) -> dict:
    """Aggregate angle statistics across all frames for LLM context.

    Args:
        frames_data: List of frame dicts with "dancer_pose" keys.
            Each frame may optionally include a "joints_3d" key with
            a (24, 3) array of SMPL joint positions from WHAM.

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
    head_lateral = []
    head_forward = []
    wrist_flexion_left = []
    wrist_flexion_right = []
    finger_ext_left = []
    finger_ext_right = []
    thumb_ext_left = []
    thumb_ext_right = []
    shoulder_elev_left = []
    shoulder_elev_right = []
    shoulder_elev_avg = []
    neck_tilt_abs = []

    # 3D angle accumulators
    knee_angle_3d = []
    left_knee_angle_3d = []
    right_knee_angle_3d = []
    torso_angle_3d = []
    arm_ext_left_3d = []
    arm_ext_right_3d = []
    hip_abd_left = []
    hip_abd_right = []
    hip_sym_3d = []
    torso_twist = []

    for frame in frames_data:
        pose = frame.get("dancer_pose", {})
        if not pose:
            continue

        angles = compute_frame_angles(
            pose,
            face=frame.get("face"),
            left_hand=frame.get("left_hand"),
            right_hand=frame.get("right_hand"),
            joints_3d=frame.get("joints_3d"),
        )

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
        if "head_lateral_tilt" in angles:
            head_lateral.append(angles["head_lateral_tilt"])
        if "head_forward_tilt" in angles:
            head_forward.append(angles["head_forward_tilt"])
        if "left_wrist_flexion" in angles:
            wrist_flexion_left.append(angles["left_wrist_flexion"])
        if "right_wrist_flexion" in angles:
            wrist_flexion_right.append(angles["right_wrist_flexion"])
        if angles.get("left_finger_extension") is not None:
            finger_ext_left.append(angles["left_finger_extension"])
        if angles.get("right_finger_extension") is not None:
            finger_ext_right.append(angles["right_finger_extension"])
        if angles.get("left_thumb_extension") is not None:
            thumb_ext_left.append(angles["left_thumb_extension"])
        if angles.get("right_thumb_extension") is not None:
            thumb_ext_right.append(angles["right_thumb_extension"])
        if "left_shoulder_elevation" in angles:
            shoulder_elev_left.append(angles["left_shoulder_elevation"])
        if "right_shoulder_elevation" in angles:
            shoulder_elev_right.append(angles["right_shoulder_elevation"])
        if "shoulder_elevation_avg" in angles:
            shoulder_elev_avg.append(angles["shoulder_elevation_avg"])
        if "neck_lateral_tilt_abs" in angles:
            neck_tilt_abs.append(angles["neck_lateral_tilt_abs"])

        # Collect 3D angles (present only when joints_3d was provided)
        if "knee_angle_3d" in angles:
            knee_angle_3d.append(angles["knee_angle_3d"])
        if "left_knee_angle_3d" in angles:
            left_knee_angle_3d.append(angles["left_knee_angle_3d"])
        if "right_knee_angle_3d" in angles:
            right_knee_angle_3d.append(angles["right_knee_angle_3d"])
        if "torso_angle_3d" in angles:
            torso_angle_3d.append(angles["torso_angle_3d"])
        if "arm_extension_left_3d" in angles:
            arm_ext_left_3d.append(angles["arm_extension_left_3d"])
        if "arm_extension_right_3d" in angles:
            arm_ext_right_3d.append(angles["arm_extension_right_3d"])
        if "hip_abduction_left" in angles:
            hip_abd_left.append(angles["hip_abduction_left"])
        if "hip_abduction_right" in angles:
            hip_abd_right.append(angles["hip_abduction_right"])
        if "hip_symmetry_3d" in angles:
            hip_sym_3d.append(angles["hip_symmetry_3d"])
        if "torso_twist" in angles:
            torso_twist.append(angles["torso_twist"])

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

    # Head tilt / shirobheda
    if head_lateral:
        summary["avg_head_lateral_tilt"] = sum(head_lateral) / len(head_lateral)
        summary["avg_head_lateral_tilt_abs"] = sum(abs(v) for v in head_lateral) / len(head_lateral)
    if head_forward:
        summary["avg_head_forward_tilt"] = sum(head_forward) / len(head_forward)

    # Wrist flexion (degrees, ~180 = straight, <180 = flexed)
    if wrist_flexion_left:
        summary["avg_wrist_flexion_left"] = sum(wrist_flexion_left) / len(wrist_flexion_left)
    if wrist_flexion_right:
        summary["avg_wrist_flexion_right"] = sum(wrist_flexion_right) / len(wrist_flexion_right)
    if wrist_flexion_left and wrist_flexion_right:
        all_wrist = wrist_flexion_left + wrist_flexion_right
        summary["avg_wrist_flexion"] = sum(all_wrist) / len(all_wrist)

    # Finger extension (degrees, ~180 = extended, ~90 = curled)
    if finger_ext_left:
        summary["avg_finger_extension_left"] = sum(finger_ext_left) / len(finger_ext_left)
    if finger_ext_right:
        summary["avg_finger_extension_right"] = sum(finger_ext_right) / len(finger_ext_right)
    if thumb_ext_left:
        summary["avg_thumb_extension_left"] = sum(thumb_ext_left) / len(thumb_ext_left)
    if thumb_ext_right:
        summary["avg_thumb_extension_right"] = sum(thumb_ext_right) / len(thumb_ext_right)

    # Shoulder elevation (higher = more relaxed, better)
    if shoulder_elev_left:
        summary["avg_shoulder_elevation_left"] = sum(shoulder_elev_left) / len(shoulder_elev_left)
    if shoulder_elev_right:
        summary["avg_shoulder_elevation_right"] = sum(shoulder_elev_right) / len(shoulder_elev_right)
    if shoulder_elev_avg:
        summary["avg_shoulder_elevation"] = sum(shoulder_elev_avg) / len(shoulder_elev_avg)

    # Neck lateral tilt / attami (degrees)
    if neck_tilt_abs:
        summary["avg_neck_lateral_tilt"] = sum(neck_tilt_abs) / len(neck_tilt_abs)
        summary["max_neck_lateral_tilt"] = max(neck_tilt_abs)

    # --- 3D angle summaries (from WHAM, when available) ---
    def _mean(lst):
        return sum(lst) / len(lst) if lst else None

    def _std(lst):
        if len(lst) < 2:
            return None
        m = sum(lst) / len(lst)
        return math.sqrt(sum((v - m) ** 2 for v in lst) / len(lst))

    if knee_angle_3d:
        summary["avg_knee_angle_3d"] = _mean(knee_angle_3d)
        summary["knee_angle_3d_std"] = _std(knee_angle_3d)
        summary["min_knee_angle_3d"] = min(knee_angle_3d)
        summary["max_knee_angle_3d"] = max(knee_angle_3d)
    if left_knee_angle_3d:
        summary["avg_left_knee_angle_3d"] = _mean(left_knee_angle_3d)
    if right_knee_angle_3d:
        summary["avg_right_knee_angle_3d"] = _mean(right_knee_angle_3d)
    if torso_angle_3d:
        summary["avg_torso_angle_3d"] = _mean(torso_angle_3d)
        summary["torso_angle_3d_std"] = _std(torso_angle_3d)
    if arm_ext_left_3d:
        summary["avg_arm_extension_left_3d"] = _mean(arm_ext_left_3d)
    if arm_ext_right_3d:
        summary["avg_arm_extension_right_3d"] = _mean(arm_ext_right_3d)
    if hip_abd_left:
        summary["avg_hip_abduction_left"] = _mean(hip_abd_left)
    if hip_abd_right:
        summary["avg_hip_abduction_right"] = _mean(hip_abd_right)
    if hip_sym_3d:
        summary["avg_hip_symmetry_3d"] = _mean(hip_sym_3d)
    if torso_twist:
        summary["avg_torso_twist"] = _mean(torso_twist)
        summary["torso_twist_std"] = _std(torso_twist)
        summary["max_torso_twist"] = max(torso_twist)

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

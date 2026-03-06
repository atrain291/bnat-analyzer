"""Compute numeric scores (0-100) from pose statistics for Bharatanatyam analysis.

Translates the raw pose statistics from summarize_pose_statistics() into
interpretable scores that populate the Analysis table.
"""


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _score_aramandi(pose_summary: dict) -> float:
    """Score aramandi (knee bend) quality, 0-100.

    Ideal knee angle is ~105 degrees.
    Linear dropoff to 0 at 45 deg (too deep) and 0 at 180 deg (straight legs).
    Consistency (low std) provides a small bonus.
    """
    avg_knee = pose_summary.get("avg_knee_angle")
    if avg_knee is None:
        return 0.0

    ideal = 105.0
    if avg_knee < ideal:
        # 105 -> 100, 45 -> 0  (range of 60 degrees below ideal)
        score = max(0.0, (avg_knee - 45.0) / (ideal - 45.0)) * 100.0
    else:
        # 105 -> 100, 180 -> 0  (range of 75 degrees above ideal)
        score = max(0.0, (180.0 - avg_knee) / (180.0 - ideal)) * 100.0

    # Penalize inconsistency: high std means the dancer isn't holding aramandi
    knee_std = pose_summary.get("knee_angle_std", 0.0)
    if knee_std is not None and knee_std > 5.0:
        # Deduct up to 20 points for high variance (std > 30 = full penalty)
        penalty = min(20.0, (knee_std - 5.0) / 25.0 * 20.0)
        score = max(0.0, score - penalty)

    return _clamp(round(score, 1))


def _score_upper_body(pose_summary: dict) -> float:
    """Score upper body uprightness, 0-100.

    Perfect upright = 0 deg deviation = 100.
    Linear dropoff: 0 at 15 degrees deviation.
    """
    avg_torso = pose_summary.get("avg_torso_angle")
    if avg_torso is None:
        return 0.0

    score = max(0.0, (15.0 - avg_torso) / 15.0) * 100.0
    return _clamp(round(score, 1))


def _score_symmetry(pose_summary: dict) -> float:
    """Score bilateral symmetry, 0-100.

    Based on hip symmetry, arm extension difference, and foot turnout difference.
    Each component: 100 at 0 deviation, 0 at threshold.
    """
    components = []

    # Hip symmetry: 100 at 0, 0 at 0.15
    hip_sym = pose_summary.get("hip_symmetry_avg")
    if hip_sym is not None:
        hip_score = max(0.0, (0.15 - hip_sym) / 0.15) * 100.0
        components.append(hip_score)

    # Arm extension symmetry: difference between left and right
    arm_left = pose_summary.get("avg_arm_extension_left")
    arm_right = pose_summary.get("avg_arm_extension_right")
    if arm_left is not None and arm_right is not None:
        arm_diff = abs(arm_left - arm_right)
        # 0 diff = 100, 30 deg diff = 0
        arm_score = max(0.0, (30.0 - arm_diff) / 30.0) * 100.0
        components.append(arm_score)

    # Foot turnout symmetry: difference between left and right
    ft_left = pose_summary.get("avg_foot_turnout_left")
    ft_right = pose_summary.get("avg_foot_turnout_right")
    if ft_left is not None and ft_right is not None:
        ft_diff = abs(ft_left - ft_right)
        # 0 diff = 100, 20 deg diff = 0
        ft_score = max(0.0, (20.0 - ft_diff) / 20.0) * 100.0
        components.append(ft_score)

    if not components:
        return 0.0

    return _clamp(round(sum(components) / len(components), 1))


def _score_foot_technique(pose_summary: dict) -> float:
    """Score foot technique, 0-100.

    Foot turnout: ideal 52.5 deg (midpoint of 45-60 range).
      100 at 52.5, 0 at 0 or 90 deg.
    Foot flatness: lower is better.
      100 at 0, 0 at 0.05.
    """
    components = []

    # Foot turnout score
    avg_turnout = pose_summary.get("avg_foot_turnout")
    if avg_turnout is not None:
        ideal_turnout = 52.5
        if avg_turnout <= ideal_turnout:
            # 0 -> 0, 52.5 -> 100
            turnout_score = (avg_turnout / ideal_turnout) * 100.0
        else:
            # 52.5 -> 100, 90 -> 0
            turnout_score = max(0.0, (90.0 - avg_turnout) / (90.0 - ideal_turnout)) * 100.0
        components.append(max(0.0, turnout_score))

    # Foot flatness score
    avg_flatness = pose_summary.get("avg_foot_flatness")
    if avg_flatness is not None:
        # 0 -> 100, 0.05 -> 0
        flatness_score = max(0.0, (0.05 - avg_flatness) / 0.05) * 100.0
        components.append(flatness_score)

    if not components:
        return 0.0

    return _clamp(round(sum(components) / len(components), 1))


def compute_scores(pose_summary: dict) -> dict:
    """Compute all scoring metrics from pose statistics.

    Args:
        pose_summary: Output of summarize_pose_statistics().

    Returns:
        Dict with aramandi_score, upper_body_score, symmetry_score,
        foot_technique_score, overall_score, and technique_scores (detailed).
    """
    if not pose_summary:
        return {
            "aramandi_score": 0.0,
            "upper_body_score": 0.0,
            "symmetry_score": 0.0,
            "foot_technique_score": 0.0,
            "overall_score": 0.0,
            "technique_scores": {},
        }

    aramandi = _score_aramandi(pose_summary)
    upper_body = _score_upper_body(pose_summary)
    symmetry = _score_symmetry(pose_summary)
    foot_technique = _score_foot_technique(pose_summary)

    # Weighted overall: aramandi 30%, upper body 20%, symmetry 25%, foot technique 25%
    overall = _clamp(round(
        aramandi * 0.30
        + upper_body * 0.20
        + symmetry * 0.25
        + foot_technique * 0.25,
        1,
    ))

    technique_scores = {
        "aramandi_score": aramandi,
        "upper_body_score": upper_body,
        "symmetry_score": symmetry,
        "foot_technique_score": foot_technique,
        "overall_score": overall,
        # Raw inputs for transparency
        "inputs": {
            "avg_knee_angle": pose_summary.get("avg_knee_angle"),
            "knee_angle_std": pose_summary.get("knee_angle_std"),
            "avg_torso_angle": pose_summary.get("avg_torso_angle"),
            "hip_symmetry_avg": pose_summary.get("hip_symmetry_avg"),
            "avg_foot_turnout": pose_summary.get("avg_foot_turnout"),
            "avg_foot_flatness": pose_summary.get("avg_foot_flatness"),
        },
    }

    return {
        "aramandi_score": aramandi,
        "upper_body_score": upper_body,
        "symmetry_score": symmetry,
        "foot_technique_score": foot_technique,
        "overall_score": overall,
        "technique_scores": technique_scores,
    }

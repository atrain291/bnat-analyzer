from app.pipeline.angles import compute_frame_angles, summarize_pose_statistics


def _make_point(x, y, confidence=0.9):
    return {"x": x, "y": y, "z": 0.0, "confidence": confidence}


def _aramandi_pose():
    """A rough aramandi pose with bent knees."""
    return {
        "nose": _make_point(0.5, 0.1),
        "left_eye": _make_point(0.48, 0.09),
        "right_eye": _make_point(0.52, 0.09),
        "left_ear": _make_point(0.46, 0.1),
        "right_ear": _make_point(0.54, 0.1),
        "left_shoulder": _make_point(0.4, 0.25),
        "right_shoulder": _make_point(0.6, 0.25),
        "left_elbow": _make_point(0.35, 0.35),
        "right_elbow": _make_point(0.65, 0.35),
        "left_wrist": _make_point(0.38, 0.40),
        "right_wrist": _make_point(0.62, 0.40),
        "left_hip": _make_point(0.45, 0.50),
        "right_hip": _make_point(0.55, 0.50),
        "left_knee": _make_point(0.35, 0.65),
        "right_knee": _make_point(0.65, 0.65),
        "left_ankle": _make_point(0.40, 0.80),
        "right_ankle": _make_point(0.60, 0.80),
    }


def test_compute_frame_angles_basic():
    pose = _aramandi_pose()
    angles = compute_frame_angles(pose)

    assert "avg_knee_angle" in angles
    assert "torso_angle" in angles
    assert "arm_extension_left" in angles
    assert "arm_extension_right" in angles
    assert "hip_symmetry" in angles

    # Knee angles should be between 0 and 180
    assert 0 < angles["avg_knee_angle"] < 180
    # Torso should be roughly upright (small angle)
    assert angles["torso_angle"] < 10
    # Hips should be roughly symmetric
    assert angles["hip_symmetry"] < 0.05


def test_compute_frame_angles_empty_pose():
    assert compute_frame_angles({}) == {}


def test_compute_frame_angles_low_confidence():
    pose = _aramandi_pose()
    pose["left_knee"] = _make_point(0.35, 0.65, confidence=0.1)
    angles = compute_frame_angles(pose)
    # Left knee angle should be None due to low confidence
    assert angles["left_knee_angle"] is None


def test_summarize_pose_statistics():
    frames = [{"dancer_pose": _aramandi_pose()} for _ in range(10)]
    summary = summarize_pose_statistics(frames)

    assert "avg_knee_angle" in summary
    assert "min_knee_angle" in summary
    assert "max_knee_angle" in summary
    assert "avg_torso_angle" in summary
    assert "balance_score" in summary
    assert 0 <= summary["balance_score"] <= 1


def test_summarize_pose_statistics_empty():
    assert summarize_pose_statistics([]) == {}
    assert summarize_pose_statistics([{"dancer_pose": {}}]) == {}

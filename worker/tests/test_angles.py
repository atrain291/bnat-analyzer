from app.pipeline.angles import compute_frame_angles, summarize_pose_statistics, OnlineAngleAccumulator


def _make_point(x, y, confidence=0.9):
    return {"x": x, "y": y, "z": 0.0, "confidence": confidence}


def _aramandi_pose():
    """A rough aramandi pose with bent knees and turned-out feet."""
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
        # Foot keypoints - feet turned out in aramandi
        "left_big_toe": _make_point(0.35, 0.83),
        "left_small_toe": _make_point(0.36, 0.82),
        "left_heel": _make_point(0.42, 0.82),
        "right_big_toe": _make_point(0.65, 0.83),
        "right_small_toe": _make_point(0.64, 0.82),
        "right_heel": _make_point(0.58, 0.82),
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


def test_compute_frame_angles_foot_turnout():
    pose = _aramandi_pose()
    angles = compute_frame_angles(pose)

    assert "left_foot_turnout" in angles
    assert "right_foot_turnout" in angles
    # Feet are turned out, so turnout should be positive
    assert angles["left_foot_turnout"] > 0
    assert angles["right_foot_turnout"] > 0


def test_compute_frame_angles_foot_flatness():
    pose = _aramandi_pose()
    angles = compute_frame_angles(pose)

    assert "left_foot_flatness" in angles
    assert "right_foot_flatness" in angles
    # Feet are roughly flat in our test pose
    assert angles["left_foot_flatness"] < 0.05
    assert angles["right_foot_flatness"] < 0.05


def test_compute_frame_angles_foot_angle():
    pose = _aramandi_pose()
    angles = compute_frame_angles(pose)

    assert "left_foot_angle" in angles
    assert "right_foot_angle" in angles
    assert 0 < angles["left_foot_angle"] < 180
    assert 0 < angles["right_foot_angle"] < 180


def test_compute_frame_angles_empty_pose():
    assert compute_frame_angles({}) == {}


def test_compute_frame_angles_low_confidence():
    pose = _aramandi_pose()
    pose["left_knee"] = _make_point(0.35, 0.65, confidence=0.1)
    angles = compute_frame_angles(pose)
    # Left knee angle should be None due to low confidence
    assert angles["left_knee_angle"] is None


def test_compute_frame_angles_no_foot_keypoints():
    """Test that body angles still work without foot keypoints (backward compat)."""
    pose = _aramandi_pose()
    # Remove foot keypoints
    for key in ("left_big_toe", "left_small_toe", "left_heel",
                "right_big_toe", "right_small_toe", "right_heel"):
        del pose[key]
    angles = compute_frame_angles(pose)

    # Body angles still computed
    assert "avg_knee_angle" in angles
    assert "torso_angle" in angles
    # Foot angles should not be present
    assert "left_foot_turnout" not in angles
    assert "right_foot_turnout" not in angles


def test_summarize_pose_statistics():
    frames = [{"dancer_pose": _aramandi_pose()} for _ in range(10)]
    summary = summarize_pose_statistics(frames)

    assert "avg_knee_angle" in summary
    assert "min_knee_angle" in summary
    assert "max_knee_angle" in summary
    assert "avg_torso_angle" in summary
    assert "balance_score" in summary
    assert 0 <= summary["balance_score"] <= 1

    # Foot statistics
    assert "avg_foot_turnout" in summary
    assert "avg_foot_turnout_left" in summary
    assert "avg_foot_turnout_right" in summary
    assert "avg_foot_flatness" in summary


def test_summarize_pose_statistics_empty():
    assert summarize_pose_statistics([]) == {}
    assert summarize_pose_statistics([{"dancer_pose": {}}]) == {}


def test_online_accumulator_matches_batch():
    """OnlineAngleAccumulator.summarize() must match summarize_pose_statistics output."""
    frames = [{"dancer_pose": _aramandi_pose()} for _ in range(20)]
    batch_summary = summarize_pose_statistics(frames)

    accum = OnlineAngleAccumulator()
    for fd in frames:
        angles = compute_frame_angles(fd["dancer_pose"])
        accum.add_frame(angles, timestamp_ms=0, pose=fd["dancer_pose"])
    online_summary = accum.summarize()

    for key in batch_summary:
        assert key in online_summary, f"Missing key in online summary: {key}"
        bv = batch_summary[key]
        ov = online_summary[key]
        if bv is None:
            assert ov is None, f"Key {key}: batch=None but online={ov}"
        else:
            assert abs(bv - ov) < 1e-9, f"Key {key}: batch={bv} online={ov}"

    for key in online_summary:
        assert key in batch_summary, f"Extra key in online summary: {key}"


def test_online_accumulator_empty():
    accum = OnlineAngleAccumulator()
    assert accum.summarize() == {}

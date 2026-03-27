import sys
from unittest.mock import MagicMock

# Mock rtmlib before importing pose module (heavy GPU dependency)
sys.modules["rtmlib"] = MagicMock()

from app.pipeline.pose import (  # noqa: E402
    _build_ffmpeg_cmd,
    _extract_pose_data,
    KEYPOINT_NAMES,
    BODY_KEYPOINT_NAMES,
    FOOT_KEYPOINT_NAMES,
    HAND_KEYPOINT_NAMES,
    TOTAL_KEYPOINTS,
)

import numpy as np  # noqa: E402


def test_keypoint_names_count():
    assert len(BODY_KEYPOINT_NAMES) == 17
    assert len(FOOT_KEYPOINT_NAMES) == 6
    assert len(KEYPOINT_NAMES) == 23  # body + feet
    assert len(HAND_KEYPOINT_NAMES) == 21


def test_total_keypoints():
    # 17 body + 6 feet + 68 face + 21 left hand + 21 right hand = 133
    assert TOTAL_KEYPOINTS == 133


def test_build_ffmpeg_cmd_h264():
    cmd, w, h, skip = _build_ffmpeg_cmd("/path/video.mp4", "h264", 1920, 1080)
    assert "-hwaccel" in cmd
    assert "cuda" in cmd
    assert "h264_cuvid" in cmd
    assert w == 1920
    assert h == 1080
    assert skip == 1


def test_build_ffmpeg_cmd_unknown_codec():
    cmd, w, h, skip = _build_ffmpeg_cmd("/path/video.mp4", "unknown", 1280, 720)
    assert "-hwaccel" not in cmd
    assert w == 1280
    assert h == 720


def test_build_ffmpeg_cmd_hevc():
    cmd, w, h, skip = _build_ffmpeg_cmd("/path/video.mp4", "hevc", 3840, 2160)
    assert "hevc_cuvid" in cmd


def test_extract_pose_data_single_person():
    """Test extraction with a single detected person."""
    keypoints = np.random.rand(1, 133, 2) * 1000
    scores = np.random.rand(1, 133) * 0.5 + 0.5  # 0.5-1.0 confidence

    result = _extract_pose_data(keypoints, scores, 1920, 1080)

    assert "dancer_pose" in result
    assert "left_hand" in result
    assert "right_hand" in result
    assert "face" in result

    # Body + feet keypoints
    assert len(result["dancer_pose"]) == 23
    for name in KEYPOINT_NAMES:
        assert name in result["dancer_pose"]
        pt = result["dancer_pose"][name]
        assert 0 <= pt["x"] <= 1
        assert 0 <= pt["y"] <= 1
        assert 0 < pt["confidence"] <= 1

    # Foot keypoints specifically
    for name in FOOT_KEYPOINT_NAMES:
        assert name in result["dancer_pose"]

    # Hand keypoints
    assert len(result["left_hand"]) == 21
    assert len(result["right_hand"]) == 21
    assert "thumb_tip" in result["left_hand"]
    assert "index_tip" in result["right_hand"]

    # Face keypoints
    assert len(result["face"]) == 68


def test_extract_pose_data_multiple_people():
    """Test that the largest person is selected."""
    keypoints = np.zeros((2, 133, 2))
    scores = np.ones((2, 133)) * 0.9

    # Person 0: small (100x100 area)
    keypoints[0, :, 0] = np.linspace(400, 500, 133)
    keypoints[0, :, 1] = np.linspace(400, 500, 133)

    # Person 1: large (800x800 area)
    keypoints[1, :, 0] = np.linspace(100, 900, 133)
    keypoints[1, :, 1] = np.linspace(100, 900, 133)

    result = _extract_pose_data(keypoints, scores, 1920, 1080)

    # Should have picked person 1 (larger)
    nose = result["dancer_pose"]["nose"]
    assert nose["x"] < 0.1  # person 1's first keypoint is near 100/1920


def test_extract_pose_data_empty():
    """Test with no detections."""
    result = _extract_pose_data(np.array([]), np.array([]), 1920, 1080)
    assert result["dancer_pose"] == {}
    assert result["left_hand"] == {}
    assert result["right_hand"] == {}
    assert result["face"] == []

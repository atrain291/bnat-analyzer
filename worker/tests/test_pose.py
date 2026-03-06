from app.pipeline.pose import _build_ffmpeg_cmd, KEYPOINT_NAMES


def test_keypoint_names_count():
    assert len(KEYPOINT_NAMES) == 17


def test_build_ffmpeg_cmd_h264():
    cmd, w, h = _build_ffmpeg_cmd("/path/video.mp4", "h264", 1920, 1080)
    assert "-hwaccel" in cmd
    assert "cuda" in cmd
    assert "h264_cuvid" in cmd
    assert w == 1920
    assert h == 1080


def test_build_ffmpeg_cmd_unknown_codec():
    cmd, w, h = _build_ffmpeg_cmd("/path/video.mp4", "unknown", 1280, 720)
    assert "-hwaccel" not in cmd
    assert w == 1280
    assert h == 720


def test_build_ffmpeg_cmd_hevc():
    cmd, w, h = _build_ffmpeg_cmd("/path/video.mp4", "hevc", 3840, 2160)
    assert "hevc_cuvid" in cmd

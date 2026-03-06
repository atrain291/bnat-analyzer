import json
from unittest.mock import patch, MagicMock

from app.pipeline.ingest import extract_metadata


FFPROBE_OUTPUT = json.dumps({
    "streams": [{
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "r_frame_rate": "30/1",
        "nb_frames": "900",
    }],
    "format": {
        "duration": "30.0",
    },
})


@patch("app.pipeline.ingest.subprocess.run")
def test_extract_metadata(mock_run):
    mock_run.return_value = MagicMock(stdout=FFPROBE_OUTPUT, returncode=0)

    result = extract_metadata("/path/to/video.mp4")

    assert result["fps"] == 30.0
    assert result["total_frames"] == 900
    assert result["width"] == 1920
    assert result["height"] == 1080
    assert result["codec"] == "h264"
    assert result["duration_ms"] == 30000
    mock_run.assert_called_once()


@patch("app.pipeline.ingest.subprocess.run")
def test_extract_metadata_no_nb_frames(mock_run):
    output = json.dumps({
        "streams": [{
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1280,
            "height": 720,
            "r_frame_rate": "24/1",
            "nb_frames": "0",
        }],
        "format": {"duration": "10.0"},
    })
    mock_run.return_value = MagicMock(stdout=output, returncode=0)

    result = extract_metadata("/path/to/video.mp4")

    assert result["fps"] == 24.0
    assert result["total_frames"] == 240  # 10s * 24fps
    assert result["duration_ms"] == 10000


@patch("app.pipeline.ingest.subprocess.run")
def test_extract_metadata_no_video_stream(mock_run):
    output = json.dumps({
        "streams": [{"codec_type": "audio"}],
        "format": {"duration": "10.0"},
    })
    mock_run.return_value = MagicMock(stdout=output, returncode=0)

    import pytest
    with pytest.raises(ValueError, match="No video stream"):
        extract_metadata("/path/to/audio.mp3")

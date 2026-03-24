import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# Codecs that browsers can't natively play — need transcoding to H.264
_NEEDS_TRANSCODE = {"hevc", "h265", "vp9", "av1"}


def ensure_browser_playable(video_path: str) -> str:
    """Transcode video to H.264+AAC if the codec isn't browser-compatible.

    Returns the (possibly new) video path. The original file is replaced in-place
    so the existing video_url/video_key remain valid.
    """
    # Probe the codec
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_entries", "stream=codec_name,codec_type",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return video_path

    probe = json.loads(result.stdout)
    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if not video_stream:
        return video_path

    codec = video_stream.get("codec_name", "h264")
    if codec not in _NEEDS_TRANSCODE:
        logger.info(f"Video codec {codec} is browser-compatible, no transcode needed")
        return video_path

    logger.info(f"Video codec is {codec}, transcoding to H.264 for browser playback...")
    out_path = video_path + ".h264.mp4"

    # Try GPU-accelerated transcode (CUVID decode + NVENC encode).
    # Use -hwaccel cuda without -hwaccel_output_format cuda so ffmpeg
    # handles 10-bit→8-bit conversion automatically before NVENC.
    nvenc_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-hwaccel", "cuda",
    ]
    cuvid_decoders = {"hevc": "hevc_cuvid", "h265": "hevc_cuvid", "vp9": "vp9_cuvid", "av1": "av1_cuvid"}
    if codec in cuvid_decoders:
        nvenc_cmd += ["-c:v", cuvid_decoders[codec]]
    nvenc_cmd += [
        "-i", video_path,
        "-pix_fmt", "yuv420p",
        "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-map", "0:v:0", "-map", "0:a:0?",
        out_path,
    ]

    result = subprocess.run(nvenc_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"GPU transcode failed, falling back to CPU: {result.stderr[:200]}")
        cpu_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", video_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-map", "0:v:0", "-map", "0:a:0?",
            out_path,
        ]
        subprocess.run(cpu_cmd, check=True)

    # Replace original with transcoded version
    os.replace(out_path, video_path)
    logger.info(f"Transcode complete, replaced {video_path}")
    return video_path


def extract_metadata(video_path: str) -> dict:
    """Extract video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    video_stream = next(
        (s for s in probe.get("streams", []) if s["codec_type"] == "video"),
        None,
    )
    if not video_stream:
        raise ValueError("No video stream found")

    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

    total_frames = int(video_stream.get("nb_frames", 0))
    if total_frames == 0:
        duration = float(probe.get("format", {}).get("duration", 0))
        total_frames = int(duration * fps)

    width = int(video_stream.get("width", 1920))
    height = int(video_stream.get("height", 1080))
    codec = video_stream.get("codec_name", "h264")
    duration_ms = int(float(probe.get("format", {}).get("duration", 0)) * 1000)

    metadata = {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "codec": codec,
        "duration_ms": duration_ms,
    }
    logger.info(f"Video metadata: {metadata}")
    return metadata

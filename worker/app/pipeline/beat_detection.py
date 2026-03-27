"""Audio onset detection and foot strike correlation for rhythm analysis.

Extracts audio from video, detects percussive onsets (nattuvangam, mridangam),
detects foot strikes from pose data, and scores synchronicity.
"""

import logging
import os
import subprocess
import tempfile

import numpy as np

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, start_ms: int = 0) -> str | None:
    """Extract audio from video to a temp WAV file.

    Args:
        start_ms: Start offset in milliseconds (skips audio before this point).

    Returns path to temp WAV file, or None if video has no audio.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
    ]

    if start_ms > 0:
        cmd += ["-ss", str(start_ms / 1000.0)]

    cmd += [
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "22050",
        "-ac", "1",
        "-y",
        tmp.name,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"No audio extracted (ffmpeg exit {result.returncode}): {result.stderr[:200]}")
        os.unlink(tmp.name)
        return None

    # Check file has actual content
    if os.path.getsize(tmp.name) < 1000:
        logger.warning("Extracted audio file too small, likely no audio stream")
        os.unlink(tmp.name)
        return None

    return tmp.name


def detect_onsets(audio_path: str) -> dict:
    """Detect percussive onsets in audio using librosa.

    Returns dict with onset_timestamps_ms, tempo_bpm, and onset_count.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=22050)

    # Onset detection with spectral flux (good for percussive instruments)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        backtrack=True,
        delta=0.07,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    onset_timestamps_ms = [int(t * 1000) for t in onset_times]

    # Estimate tempo
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=512)
    tempo_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    logger.info(f"Onset detection: {len(onset_timestamps_ms)} onsets, tempo ~{tempo_bpm:.1f} BPM")

    return {
        "onset_timestamps_ms": onset_timestamps_ms,
        "tempo_bpm": round(tempo_bpm, 1),
        "onset_count": len(onset_timestamps_ms),
    }


def detect_foot_strikes(
    frames_data: list[dict],
    fps: float,
    min_gap_ms: int = 100,
) -> list[int]:
    """Detect foot strike events from per-frame foot flatness data.

    A foot strike is detected when foot_flatness decreases sharply
    (foot transitioning from lifted to flat on ground).

    Returns list of timestamps (ms) where foot strikes occur.
    """
    if not frames_data:
        return []

    # Collect per-frame flatness (min of left/right = whichever foot is more grounded)
    timestamps = []
    flatness_values = []

    for fd in frames_data:
        pose = fd.get("dancer_pose", {})
        if not pose:
            continue

        left_heel = pose.get("left_heel", {})
        left_toe = pose.get("left_big_toe", {})
        right_heel = pose.get("right_heel", {})
        right_toe = pose.get("right_big_toe", {})

        left_flat = None
        right_flat = None

        if left_heel.get("confidence", 0) > 0.3 and left_toe.get("confidence", 0) > 0.3:
            left_flat = abs(left_heel["y"] - left_toe["y"])
        if right_heel.get("confidence", 0) > 0.3 and right_toe.get("confidence", 0) > 0.3:
            right_flat = abs(right_heel["y"] - right_toe["y"])

        # Use minimum flatness (most grounded foot)
        if left_flat is not None and right_flat is not None:
            flat = min(left_flat, right_flat)
        elif left_flat is not None:
            flat = left_flat
        elif right_flat is not None:
            flat = right_flat
        else:
            continue

        timestamps.append(fd.get("timestamp_ms", 0))
        flatness_values.append(flat)

    if len(flatness_values) < 5:
        return []

    arr = np.array(flatness_values)

    # Smooth with 3-frame moving average to reduce pose jitter
    kernel = np.ones(3) / 3
    smoothed = np.convolve(arr, kernel, mode="same")

    # Compute first derivative (negative = foot becoming flatter = strike)
    derivative = np.diff(smoothed)

    if len(derivative) == 0 or np.std(derivative) < 1e-8:
        return []

    # Adaptive threshold: frames where derivative is strongly negative
    threshold = -np.std(derivative) * 1.5
    strike_indices = np.where(derivative < threshold)[0]

    if len(strike_indices) == 0:
        return []

    # Enforce minimum inter-strike gap
    strikes = [strike_indices[0]]
    for idx in strike_indices[1:]:
        gap = timestamps[idx] - timestamps[strikes[-1]] if idx < len(timestamps) else 0
        if gap >= min_gap_ms:
            strikes.append(idx)

    strike_timestamps = [timestamps[i] for i in strikes if i < len(timestamps)]
    logger.info(f"Foot strike detection: {len(strike_timestamps)} strikes from {len(flatness_values)} frames")
    return strike_timestamps


def detect_foot_strikes_from_series(
    timestamps: list[int],
    flatness_values: list[float],
    min_gap_ms: int = 100,
) -> list[int]:
    """Detect foot strikes from pre-extracted foot flatness time series.

    Same algorithm as detect_foot_strikes but accepts pre-collected data
    (from OnlineAngleAccumulator) instead of iterating frames_data.
    """
    if len(flatness_values) < 5:
        return []

    arr = np.array(flatness_values)

    kernel = np.ones(3) / 3
    smoothed = np.convolve(arr, kernel, mode="same")

    derivative = np.diff(smoothed)

    if len(derivative) == 0 or np.std(derivative) < 1e-8:
        return []

    threshold = -np.std(derivative) * 1.5
    strike_indices = np.where(derivative < threshold)[0]

    if len(strike_indices) == 0:
        return []

    strikes = [strike_indices[0]]
    for idx in strike_indices[1:]:
        gap = timestamps[idx] - timestamps[strikes[-1]] if idx < len(timestamps) else 0
        if gap >= min_gap_ms:
            strikes.append(idx)

    strike_timestamps = [timestamps[i] for i in strikes if i < len(timestamps)]
    logger.info(f"Foot strike detection: {len(strike_timestamps)} strikes from {len(flatness_values)} frames")
    return strike_timestamps


def score_rhythm_sync(
    onset_timestamps_ms: list[int],
    strike_timestamps_ms: list[int],
    tolerance_ms: int = 75,
) -> dict:
    """Score synchronicity between audio onsets and foot strikes.

    Returns dict with rhythm_score (0-100), match_rate, avg_offset_ms,
    and per-strike details.
    """
    if not onset_timestamps_ms or not strike_timestamps_ms:
        return {
            "rhythm_score": None,
            "match_rate": 0.0,
            "avg_offset_ms": None,
            "matched_strikes": 0,
            "total_strikes": len(strike_timestamps_ms),
        }

    onset_arr = np.array(onset_timestamps_ms)

    matched_count = 0
    total_offset = 0.0

    for strike_t in strike_timestamps_ms:
        distances = np.abs(onset_arr - strike_t)
        nearest = np.min(distances)
        if nearest <= tolerance_ms:
            matched_count += 1
            total_offset += nearest

    if matched_count == 0:
        return {
            "rhythm_score": 0.0,
            "match_rate": 0.0,
            "avg_offset_ms": None,
            "matched_strikes": 0,
            "total_strikes": len(strike_timestamps_ms),
        }

    match_rate = matched_count / len(strike_timestamps_ms)
    avg_offset = total_offset / matched_count
    precision = 1.0 - (avg_offset / tolerance_ms)

    # 60% match rate + 40% precision, scaled 0-100
    score = (match_rate * 0.6 + precision * 0.4) * 100.0
    score = max(0.0, min(100.0, round(score, 1)))

    logger.info(
        f"Rhythm sync: {score}/100 "
        f"({matched_count}/{len(strike_timestamps_ms)} strikes matched, "
        f"avg offset {avg_offset:.1f}ms)"
    )

    return {
        "rhythm_score": score,
        "match_rate": round(match_rate, 3),
        "avg_offset_ms": round(avg_offset, 1),
        "matched_strikes": matched_count,
        "total_strikes": len(strike_timestamps_ms),
    }


def run_beat_analysis(video_path: str, metadata: dict, start_ms: int = 0) -> dict | None:
    """Run full audio onset detection pipeline.

    Args:
        start_ms: Start offset in milliseconds (skips audio before this point).

    Returns dict with onset data, or None if no audio available.
    """
    audio_path = extract_audio(video_path, start_ms=start_ms)
    if not audio_path:
        logger.info("No audio stream, skipping beat analysis")
        return None

    try:
        result = detect_onsets(audio_path)
        return result
    except Exception as e:
        logger.warning(f"Beat detection failed: {e}")
        return None
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

"""Audio cross-correlation for temporal synchronization of multi-angle videos.

Extracts audio from two videos, computes cross-correlation to find the
time offset that aligns them. No camera geometry needed — just audio.
"""

import logging
import os
import subprocess
import tempfile

import numpy as np

logger = logging.getLogger(__name__)


def extract_audio_mono(video_path: str, sr: int = 16000) -> tuple[np.ndarray, int] | None:
    """Extract audio from video as mono PCM at given sample rate.

    Returns (samples_array, sample_rate) or None if no audio.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        "-f", "s16le",
        "-y",
        tmp.name,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"No audio extracted from {video_path}: {result.stderr[:200]}")
        os.unlink(tmp.name)
        return None

    if os.path.getsize(tmp.name) < 1000:
        logger.warning(f"Audio too small from {video_path}")
        os.unlink(tmp.name)
        return None

    samples = np.fromfile(tmp.name, dtype=np.int16).astype(np.float32)
    os.unlink(tmp.name)

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val

    return samples, sr


def compute_sync_offset(
    video_path_a: str,
    video_path_b: str,
    sr: int = 16000,
    max_offset_sec: float = 30.0,
) -> dict:
    """Compute the time offset between two videos using audio cross-correlation.

    Returns dict with:
        offset_ms: How many ms video_b is ahead of video_a.
                   Positive = B starts later, negative = B starts earlier.
        confidence: 0-1 correlation strength.
        method: "audio_cross_correlation"

    If audio extraction fails for either video, returns offset_ms=0 with low confidence.
    """
    audio_a = extract_audio_mono(video_path_a, sr)
    audio_b = extract_audio_mono(video_path_b, sr)

    if audio_a is None or audio_b is None:
        logger.warning("Could not extract audio from one or both videos, assuming zero offset")
        return {
            "offset_ms": 0,
            "confidence": 0.0,
            "method": "no_audio",
        }

    samples_a, _ = audio_a
    samples_b, _ = audio_b

    # Limit cross-correlation search range
    max_lag = int(max_offset_sec * sr)

    # Use shorter signal for efficiency
    if len(samples_a) > len(samples_b):
        samples_a, samples_b = samples_b, samples_a
        swapped = True
    else:
        swapped = False

    # Truncate to reasonable length (first 60 seconds is usually enough)
    max_samples = 60 * sr
    a = samples_a[:max_samples]
    b = samples_b[:max_samples]

    # Cross-correlate using FFT for efficiency
    n = len(a) + len(b) - 1
    # Pad to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < n:
        fft_size *= 2

    fft_a = np.fft.rfft(a, fft_size)
    fft_b = np.fft.rfft(b, fft_size)
    cross_corr = np.fft.irfft(fft_a * np.conj(fft_b), fft_size)

    # The correlation output wraps around — rearrange to center
    # Positive lags (b is delayed) are at the beginning
    # Negative lags (a is delayed) are at the end
    half = fft_size // 2

    # Only search within max_lag range
    search_range = min(max_lag, half)

    # Positive lags: cross_corr[0:search_range]
    # Negative lags: cross_corr[-search_range:]
    positive_lags = cross_corr[:search_range]
    negative_lags = cross_corr[-search_range:]

    # Combine into a searchable array
    search_window = np.concatenate([negative_lags, positive_lags])
    peak_idx = np.argmax(np.abs(search_window))

    # Convert index to lag in samples
    lag_samples = peak_idx - search_range
    if swapped:
        lag_samples = -lag_samples

    offset_ms = int(lag_samples * 1000 / sr)

    # Compute confidence as normalized peak correlation
    peak_val = np.abs(search_window[peak_idx])
    norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    confidence = float(peak_val / norm) if norm > 0 else 0.0
    confidence = min(1.0, confidence)

    logger.info(
        f"Audio sync: offset={offset_ms}ms, confidence={confidence:.3f}, "
        f"lag_samples={lag_samples}"
    )

    return {
        "offset_ms": offset_ms,
        "confidence": round(confidence, 4),
        "method": "audio_cross_correlation",
    }


def compute_group_sync_offsets(video_paths: dict[int, str]) -> dict:
    """Compute sync offsets for a group of videos relative to the first.

    Args:
        video_paths: {performance_id: video_path}

    Returns dict with:
        offsets: {performance_id: offset_ms} (first video is reference = 0)
        confidence: minimum confidence across all pairs
    """
    perf_ids = sorted(video_paths.keys())
    if len(perf_ids) < 2:
        return {"offsets": {perf_ids[0]: 0} if perf_ids else {}, "confidence": 1.0}

    reference_id = perf_ids[0]
    offsets = {reference_id: 0}
    min_confidence = 1.0

    for perf_id in perf_ids[1:]:
        result = compute_sync_offset(
            video_paths[reference_id],
            video_paths[perf_id],
        )
        offsets[perf_id] = result["offset_ms"]
        min_confidence = min(min_confidence, result["confidence"])

    logger.info(f"Group sync offsets: {offsets}, min_confidence={min_confidence:.3f}")

    return {
        "offsets": offsets,
        "confidence": round(min_confidence, 4),
    }

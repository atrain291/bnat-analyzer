"""Extract appearance descriptors (dominant colors) from person bounding boxes."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

# HSV-based color naming: (hue_min, hue_max, sat_min, val_min) -> name
# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255
_COLOR_RULES = [
    # Achromatic colors (low saturation)
    {"name": "white", "h": (0, 180), "s": (0, 40), "v": (200, 256)},
    {"name": "gray", "h": (0, 180), "s": (0, 40), "v": (80, 200)},
    {"name": "black", "h": (0, 180), "s": (0, 40), "v": (0, 80)},
    # Chromatic colors
    {"name": "red", "h": (0, 10), "s": (60, 256), "v": (50, 256)},
    {"name": "red", "h": (170, 180), "s": (60, 256), "v": (50, 256)},
    {"name": "orange", "h": (10, 22), "s": (60, 256), "v": (50, 256)},
    {"name": "yellow", "h": (22, 35), "s": (60, 256), "v": (50, 256)},
    {"name": "green", "h": (35, 80), "s": (40, 256), "v": (40, 256)},
    {"name": "teal", "h": (80, 95), "s": (40, 256), "v": (40, 256)},
    {"name": "blue", "h": (95, 130), "s": (40, 256), "v": (40, 256)},
    {"name": "purple", "h": (130, 155), "s": (40, 256), "v": (40, 256)},
    {"name": "pink", "h": (155, 170), "s": (40, 256), "v": (50, 256)},
]


def _classify_pixel_color(h: int, s: int, v: int) -> str:
    """Map a single HSV pixel to a color name."""
    for rule in _COLOR_RULES:
        h_lo, h_hi = rule["h"]
        s_lo, s_hi = rule["s"]
        v_lo, v_hi = rule["v"]
        if h_lo <= h < h_hi and s_lo <= s < s_hi and v_lo <= v < v_hi:
            return rule["name"]
    return "unknown"


def extract_appearance(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    normalized: bool = True,
) -> dict:
    """Extract appearance descriptor from a person's bounding box region.

    Args:
        frame_bgr: Full frame in BGR format (H, W, 3).
        bbox: (x_min, y_min, x_max, y_max). If normalized=True, values are 0-1.

    Returns:
        {
            "dominant_colors": [{"name": "red", "rgb": [200, 40, 30], "pct": 0.45}, ...],
            "color_histogram": [int, ...],  # 72-bin HSV histogram for tracker
            "description": "red top"  # human-readable summary
        }
    """
    import cv2

    h_frame, w_frame = frame_bgr.shape[:2]

    if normalized:
        x1 = int(bbox[0] * w_frame)
        y1 = int(bbox[1] * h_frame)
        x2 = int(bbox[2] * w_frame)
        y2 = int(bbox[3] * h_frame)
    else:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Clamp to frame bounds
    x1 = max(0, min(x1, w_frame - 1))
    x2 = max(x1 + 1, min(x2, w_frame))
    y1 = max(0, min(y1, h_frame - 1))
    y2 = max(y1 + 1, min(y2, h_frame))

    # Focus on upper 60% of bbox (torso region — more discriminative than legs)
    torso_y2 = y1 + int((y2 - y1) * 0.6)
    # Also trim 10% on each side to avoid background pixels
    margin_x = int((x2 - x1) * 0.1)
    crop = frame_bgr[y1:torso_y2, x1 + margin_x:x2 - margin_x]

    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return {"dominant_colors": [], "color_histogram": [], "description": "unknown"}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Compute 72-bin histogram: 8 hue × 3 saturation × 3 value
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [8, 3, 3], [0, 180, 0, 256, 0, 256]
    )
    hist = hist.flatten()
    total = hist.sum()
    if total > 0:
        hist = hist / total
    color_histogram = [round(float(v), 4) for v in hist]

    # Extract dominant colors by classifying a sample of pixels
    pixels = hsv.reshape(-1, 3)
    # Sample up to 500 pixels for speed
    if len(pixels) > 500:
        indices = np.random.default_rng(42).choice(len(pixels), 500, replace=False)
        pixels = pixels[indices]

    color_counts: dict[str, list] = {}
    for px in pixels:
        name = _classify_pixel_color(int(px[0]), int(px[1]), int(px[2]))
        if name == "unknown":
            continue
        if name not in color_counts:
            color_counts[name] = []
        color_counts[name].append(px)

    total_classified = sum(len(v) for v in color_counts.values())
    if total_classified == 0:
        return {"dominant_colors": [], "color_histogram": color_histogram, "description": "unknown"}

    # Sort by frequency, keep top 3
    sorted_colors = sorted(color_counts.items(), key=lambda x: -len(x[1]))[:3]

    dominant_colors = []
    for name, px_list in sorted_colors:
        pct = len(px_list) / total_classified
        if pct < 0.08:
            continue
        # Average HSV -> convert to RGB for display
        avg_hsv = np.mean(px_list, axis=0).astype(np.uint8).reshape(1, 1, 3)
        avg_rgb = cv2.cvtColor(avg_hsv, cv2.COLOR_HSV2RGB)[0, 0]
        dominant_colors.append({
            "name": name,
            "rgb": [int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2])],
            "pct": round(pct, 2),
        })

    # Human-readable description: "red and black top" or "blue top"
    color_names = [c["name"] for c in dominant_colors[:2]]
    if color_names:
        description = " and ".join(color_names) + " top"
    else:
        description = "unknown"

    return {
        "dominant_colors": dominant_colors,
        "color_histogram": color_histogram,
        "description": description,
    }


def histogram_similarity(hist_a: list[float], hist_b: list[float]) -> float:
    """Compute similarity between two color histograms (0.0 = different, 1.0 = identical).

    Uses histogram intersection (Swain & Ballard), which is robust and fast.
    """
    if not hist_a or not hist_b or len(hist_a) != len(hist_b):
        return 0.0
    return float(sum(min(a, b) for a, b in zip(hist_a, hist_b)))

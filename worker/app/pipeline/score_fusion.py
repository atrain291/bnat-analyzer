"""Multi-angle score fusion: combine per-view scores into consensus scores.

Different camera angles provide better visibility for different metrics:
- Front view: symmetry, arm extension, foot turnout
- Side view: torso uprightness, aramandi depth, forward lean
- 3/4 view: balanced visibility

When camera labels are known, we weight scores by view reliability.
Otherwise, we average across views.
"""

import logging

logger = logging.getLogger(__name__)

# Weight multipliers by camera angle and metric.
# Higher weight = this angle is more reliable for this metric.
VIEW_WEIGHTS = {
    "front": {
        "aramandi_score": 0.7,
        "upper_body_score": 0.5,
        "symmetry_score": 1.0,
        "foot_technique_score": 0.8,
        "rhythm_consistency_score": 1.0,
    },
    "side": {
        "aramandi_score": 1.0,
        "upper_body_score": 1.0,
        "symmetry_score": 0.4,
        "foot_technique_score": 0.6,
        "rhythm_consistency_score": 1.0,
    },
    "3/4": {
        "aramandi_score": 0.85,
        "upper_body_score": 0.8,
        "symmetry_score": 0.7,
        "foot_technique_score": 0.7,
        "rhythm_consistency_score": 1.0,
    },
}

SCORE_FIELDS = [
    "aramandi_score",
    "upper_body_score",
    "symmetry_score",
    "rhythm_consistency_score",
    "overall_score",
]


def _normalize_camera_label(label: str | None) -> str:
    """Normalize camera label to a known category for weight lookup."""
    if not label:
        return "unknown"
    label_lower = label.lower().strip()
    if "front" in label_lower:
        return "front"
    if "side" in label_lower:
        return "side"
    if "3/4" in label_lower or "three" in label_lower or "quarter" in label_lower:
        return "3/4"
    return "unknown"


def fuse_scores(
    per_view_analyses: list[dict],
) -> dict:
    """Fuse scores from multiple camera views into consensus scores.

    Args:
        per_view_analyses: List of dicts, each with:
            - performance_id: int
            - camera_label: str | None
            - aramandi_score, upper_body_score, symmetry_score,
              rhythm_consistency_score, overall_score: float | None
            - technique_scores: dict | None

    Returns dict with:
        - Fused scores for each metric
        - per_view_scores: {performance_id: {metric: score}}
        - score_sources: {metric: performance_id} (which view was weighted most)
    """
    if not per_view_analyses:
        return {
            "aramandi_score": None,
            "upper_body_score": None,
            "symmetry_score": None,
            "rhythm_consistency_score": None,
            "overall_score": None,
            "per_view_scores": {},
            "score_sources": {},
        }

    if len(per_view_analyses) == 1:
        a = per_view_analyses[0]
        return {
            "aramandi_score": a.get("aramandi_score"),
            "upper_body_score": a.get("upper_body_score"),
            "symmetry_score": a.get("symmetry_score"),
            "rhythm_consistency_score": a.get("rhythm_consistency_score"),
            "overall_score": a.get("overall_score"),
            "per_view_scores": {a["performance_id"]: {f: a.get(f) for f in SCORE_FIELDS}},
            "score_sources": {f: a["performance_id"] for f in SCORE_FIELDS},
        }

    # Build per-view scores dict
    per_view_scores = {}
    for a in per_view_analyses:
        per_view_scores[a["performance_id"]] = {f: a.get(f) for f in SCORE_FIELDS}

    # Compute weighted average for each metric
    fused = {}
    score_sources = {}

    for field in SCORE_FIELDS:
        if field == "overall_score":
            continue  # Recompute from fused component scores

        weighted_sum = 0.0
        weight_sum = 0.0
        best_weight = 0.0
        best_perf_id = None

        for a in per_view_analyses:
            score = a.get(field)
            if score is None:
                continue

            cam = _normalize_camera_label(a.get("camera_label"))
            weight = VIEW_WEIGHTS.get(cam, {}).get(field, 0.75)  # default weight for unknown angles

            weighted_sum += score * weight
            weight_sum += weight

            if weight > best_weight:
                best_weight = weight
                best_perf_id = a["performance_id"]

        if weight_sum > 0:
            fused[field] = round(weighted_sum / weight_sum, 1)
            score_sources[field] = best_perf_id
        else:
            fused[field] = None

    # Recompute overall from fused components (same weights as scoring.py)
    components = []
    weights = [
        ("aramandi_score", 0.30),
        ("upper_body_score", 0.20),
        ("symmetry_score", 0.25),
    ]
    # Use rhythm if available, otherwise redistribute to foot_technique
    if fused.get("rhythm_consistency_score") is not None:
        weights.append(("rhythm_consistency_score", 0.25))
    else:
        weights = [
            ("aramandi_score", 0.30),
            ("upper_body_score", 0.20),
            ("symmetry_score", 0.25),
        ]

    total_weight = 0.0
    overall = 0.0
    for field, w in weights:
        if fused.get(field) is not None:
            overall += fused[field] * w
            total_weight += w

    if total_weight > 0:
        fused["overall_score"] = round(overall / total_weight * sum(w for _, w in weights), 1)
        fused["overall_score"] = max(0.0, min(100.0, fused["overall_score"]))
    else:
        fused["overall_score"] = None

    score_sources["overall_score"] = "fused"

    logger.info(f"Score fusion: {fused}")

    return {
        **fused,
        "per_view_scores": per_view_scores,
        "score_sources": score_sources,
    }

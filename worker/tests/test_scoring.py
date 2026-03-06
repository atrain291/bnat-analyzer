"""Tests for the scoring module (worker/app/pipeline/scoring.py)."""

import pytest

from app.pipeline.scoring import compute_scores


# ---------------------------------------------------------------------------
# Fixtures: representative pose summaries
# ---------------------------------------------------------------------------

def _ideal_pose_summary():
    """Near-perfect Bharatanatyam pose statistics."""
    return {
        "avg_knee_angle": 105.0,       # ideal aramandi
        "min_knee_angle": 100.0,
        "max_knee_angle": 110.0,
        "knee_angle_std": 3.0,         # very consistent
        "avg_torso_angle": 1.0,        # nearly upright
        "avg_arm_extension_left": 160.0,
        "avg_arm_extension_right": 160.0,  # symmetric arms
        "hip_symmetry_avg": 0.01,      # nearly level hips
        "balance_score": 0.95,
        "avg_foot_turnout": 52.5,      # ideal midpoint
        "avg_foot_turnout_left": 52.0,
        "avg_foot_turnout_right": 53.0,
        "avg_foot_flatness": 0.005,    # very flat feet
    }


def _poor_pose_summary():
    """Poor posture: straight legs, leaning torso, asymmetric."""
    return {
        "avg_knee_angle": 170.0,       # nearly straight legs
        "min_knee_angle": 160.0,
        "max_knee_angle": 178.0,
        "knee_angle_std": 15.0,        # inconsistent
        "avg_torso_angle": 14.0,       # heavy lean
        "avg_arm_extension_left": 90.0,
        "avg_arm_extension_right": 150.0,  # big asymmetry
        "hip_symmetry_avg": 0.14,      # very uneven hips
        "balance_score": 0.2,
        "avg_foot_turnout": 10.0,      # feet barely turned out
        "avg_foot_turnout_left": 5.0,
        "avg_foot_turnout_right": 15.0,
        "avg_foot_flatness": 0.045,    # on toes
    }


# ---------------------------------------------------------------------------
# Test: ideal pose should score high
# ---------------------------------------------------------------------------

class TestIdealPose:
    def test_aramandi_high(self):
        scores = compute_scores(_ideal_pose_summary())
        assert scores["aramandi_score"] >= 90.0

    def test_upper_body_high(self):
        scores = compute_scores(_ideal_pose_summary())
        assert scores["upper_body_score"] >= 90.0

    def test_symmetry_high(self):
        scores = compute_scores(_ideal_pose_summary())
        assert scores["symmetry_score"] >= 85.0

    def test_foot_technique_high(self):
        scores = compute_scores(_ideal_pose_summary())
        assert scores["foot_technique_score"] >= 90.0

    def test_overall_high(self):
        scores = compute_scores(_ideal_pose_summary())
        assert scores["overall_score"] >= 85.0

    def test_technique_scores_present(self):
        scores = compute_scores(_ideal_pose_summary())
        ts = scores["technique_scores"]
        assert "aramandi_score" in ts
        assert "upper_body_score" in ts
        assert "symmetry_score" in ts
        assert "foot_technique_score" in ts
        assert "overall_score" in ts
        assert "inputs" in ts


# ---------------------------------------------------------------------------
# Test: poor posture should score low
# ---------------------------------------------------------------------------

class TestPoorPose:
    def test_aramandi_low(self):
        scores = compute_scores(_poor_pose_summary())
        assert scores["aramandi_score"] <= 25.0

    def test_upper_body_low(self):
        scores = compute_scores(_poor_pose_summary())
        assert scores["upper_body_score"] <= 15.0

    def test_symmetry_low(self):
        scores = compute_scores(_poor_pose_summary())
        assert scores["symmetry_score"] <= 30.0

    def test_foot_technique_low(self):
        scores = compute_scores(_poor_pose_summary())
        assert scores["foot_technique_score"] <= 20.0

    def test_overall_low(self):
        scores = compute_scores(_poor_pose_summary())
        assert scores["overall_score"] <= 25.0


# ---------------------------------------------------------------------------
# Test: empty / missing stats should handle gracefully
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dict(self):
        scores = compute_scores({})
        assert scores["aramandi_score"] == 0.0
        assert scores["upper_body_score"] == 0.0
        assert scores["symmetry_score"] == 0.0
        assert scores["foot_technique_score"] == 0.0
        assert scores["overall_score"] == 0.0
        assert scores["technique_scores"] == {}

    def test_none_input(self):
        scores = compute_scores(None)
        assert scores["overall_score"] == 0.0

    def test_partial_data_only_knee(self):
        scores = compute_scores({"avg_knee_angle": 105.0})
        assert scores["aramandi_score"] > 0
        assert scores["upper_body_score"] == 0.0
        assert scores["symmetry_score"] == 0.0

    def test_all_scores_in_range(self):
        """Every score must be between 0 and 100 regardless of input."""
        for summary in [_ideal_pose_summary(), _poor_pose_summary(), {}, {"avg_knee_angle": 200}]:
            scores = compute_scores(summary)
            for key in ["aramandi_score", "upper_body_score", "symmetry_score",
                        "foot_technique_score", "overall_score"]:
                assert 0.0 <= scores[key] <= 100.0, f"{key}={scores[key]} out of range"

    def test_extreme_knee_angle(self):
        """Knee angle far outside valid range should clamp to 0."""
        scores = compute_scores({"avg_knee_angle": 200.0})
        assert scores["aramandi_score"] == 0.0

    def test_negative_torso_angle(self):
        """Negative torso angle (shouldn't happen but handle gracefully)."""
        scores = compute_scores({"avg_torso_angle": -5.0})
        # Should clamp to 100 since abs is small
        assert scores["upper_body_score"] == 100.0


# ---------------------------------------------------------------------------
# Test: overall score is correctly weighted
# ---------------------------------------------------------------------------

class TestWeighting:
    def test_overall_is_weighted_average(self):
        """Overall score = 0.30*aramandi + 0.20*upper + 0.25*symmetry + 0.25*foot."""
        scores = compute_scores(_ideal_pose_summary())
        expected = round(
            scores["aramandi_score"] * 0.30
            + scores["upper_body_score"] * 0.20
            + scores["symmetry_score"] * 0.25
            + scores["foot_technique_score"] * 0.25,
            1,
        )
        # Allow small floating-point rounding tolerance
        assert abs(scores["overall_score"] - expected) <= 0.2

    def test_overall_weighted_poor(self):
        scores = compute_scores(_poor_pose_summary())
        expected = round(
            scores["aramandi_score"] * 0.30
            + scores["upper_body_score"] * 0.20
            + scores["symmetry_score"] * 0.25
            + scores["foot_technique_score"] * 0.25,
            1,
        )
        assert abs(scores["overall_score"] - expected) <= 0.2

    def test_only_aramandi_contributes(self):
        """When only knee data is available, overall = 0.30 * aramandi."""
        scores = compute_scores({"avg_knee_angle": 105.0})
        expected = round(scores["aramandi_score"] * 0.30, 1)
        assert abs(scores["overall_score"] - expected) <= 0.2

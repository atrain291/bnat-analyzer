"""Tests for tracker module — tunable thresholds and core matching."""

import os
import sys
import types
import pytest

# Mock onnxruntime before importing tracker (GPU-only dependency)
if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from app.pipeline.tracker import SimpleTracker, compute_iou, centroid_distance


# ---------------------------------------------------------------------------
# Tunable threshold defaults
# ---------------------------------------------------------------------------

class TestThresholdDefaults:
    def test_default_values(self):
        t = SimpleTracker()
        assert t.reseed_identity_threshold == 0.4
        assert t.coherence_threshold == 0.4
        assert t.formation_ratio_limit == 3.0
        assert t.reid_base_threshold == 0.5

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TRACKER_COHERENCE_THRESHOLD", "0.6")
        monkeypatch.setenv("TRACKER_FORMATION_RATIO_LIMIT", "2.5")
        monkeypatch.setenv("TRACKER_RESEED_IDENTITY_THRESHOLD", "0.35")
        monkeypatch.setenv("TRACKER_REID_BASE_THRESHOLD", "0.45")
        # Need to reimport to pick up env vars
        import importlib
        import app.pipeline.pose_config as pc
        importlib.reload(pc)
        import app.pipeline.tracker as tracker_mod
        importlib.reload(tracker_mod)
        t = tracker_mod.SimpleTracker()
        assert t.coherence_threshold == 0.6
        assert t.formation_ratio_limit == 2.5
        assert t.reseed_identity_threshold == 0.35
        assert t.reid_base_threshold == 0.45
        # Restore defaults
        monkeypatch.delenv("TRACKER_COHERENCE_THRESHOLD")
        monkeypatch.delenv("TRACKER_FORMATION_RATIO_LIMIT")
        monkeypatch.delenv("TRACKER_RESEED_IDENTITY_THRESHOLD")
        monkeypatch.delenv("TRACKER_REID_BASE_THRESHOLD")
        importlib.reload(pc)
        importlib.reload(tracker_mod)


# ---------------------------------------------------------------------------
# Group coherence uses tunable formation_ratio_limit
# ---------------------------------------------------------------------------

class TestFormationRatioLimit:
    def test_coherence_passes_within_limit(self):
        t = SimpleTracker()
        # Seed with two tracks
        t.seed({0: (0.1, 0.1, 0.3, 0.5), 1: (0.5, 0.1, 0.7, 0.5)})
        # Candidate at similar relative distance should pass
        proposed = {1: (0.5, 0.1, 0.7, 0.5)}
        score = t._group_coherence_score((0.1, 0.1, 0.3, 0.5), 0, proposed)
        assert score == 1.0

    def test_coherence_rejects_beyond_limit(self):
        t = SimpleTracker()
        t.formation_ratio_limit = 1.5  # tighter limit for test
        t.seed({0: (0.1, 0.1, 0.3, 0.5), 1: (0.5, 0.1, 0.7, 0.5)})
        # Candidate very far away — should break formation beyond 1.5x
        proposed = {1: (0.5, 0.1, 0.7, 0.5)}
        score = t._group_coherence_score((0.9, 0.9, 1.0, 1.0), 0, proposed)
        assert score < 1.0

    def test_tighter_limit_rejects_more(self):
        t_tight = SimpleTracker()
        t_tight.formation_ratio_limit = 1.5
        t_tight.seed({0: (0.1, 0.1, 0.3, 0.5), 1: (0.5, 0.1, 0.7, 0.5)})
        t_wide = SimpleTracker()
        t_wide.formation_ratio_limit = 5.0
        t_wide.seed({0: (0.1, 0.1, 0.3, 0.5), 1: (0.5, 0.1, 0.7, 0.5)})
        # Candidate at moderate offset
        proposed = {1: (0.5, 0.1, 0.7, 0.5)}
        candidate = (0.05, 0.05, 0.25, 0.45)  # shifted but not crazy
        score_tight = t_tight._group_coherence_score(candidate, 0, proposed)
        score_wide = t_wide._group_coherence_score(candidate, 0, proposed)
        assert score_tight <= score_wide


# ---------------------------------------------------------------------------
# Basic tracker operations (regression)
# ---------------------------------------------------------------------------

class TestTrackerBasics:
    def test_first_frame_assigns_ids(self):
        t = SimpleTracker()
        bboxes = [(0.1, 0.1, 0.3, 0.5), (0.5, 0.1, 0.7, 0.5)]
        ids = t.update(bboxes)
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_consistent_tracking(self):
        t = SimpleTracker()
        bboxes = [(0.1, 0.1, 0.3, 0.5)]
        ids1 = t.update(bboxes)
        ids2 = t.update(bboxes)
        assert ids1[0] == ids2[0]

    def test_empty_frame(self):
        t = SimpleTracker()
        ids = t.update([])
        assert ids == []

    def test_seed_sets_group_ids(self):
        t = SimpleTracker()
        t.seed({0: (0.1, 0.1, 0.3, 0.5), 1: (0.5, 0.1, 0.7, 0.5)})
        assert t.group_ids == {0, 1}

    def test_iou_computation(self):
        assert compute_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
        assert compute_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
        iou = compute_iou((0, 0, 2, 2), (1, 1, 3, 3))
        assert 0.1 < iou < 0.2

    def test_centroid_distance(self):
        assert centroid_distance((0, 0, 2, 2), (0, 0, 2, 2)) == 0.0
        d = centroid_distance((0, 0, 2, 2), (2, 0, 4, 2))
        assert abs(d - 2.0) < 1e-6

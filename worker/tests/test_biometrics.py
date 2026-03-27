"""Tests for biometrics module — 2D and 3D signature extraction."""

import pytest

from app.pipeline.biometrics import (
    BiometricSignature,
    extract_biometric_signature,
    extract_biometric_signature_3d,
    merge_signatures,
    signature_similarity,
)


# ---------------------------------------------------------------------------
# Helpers: realistic joint positions
# ---------------------------------------------------------------------------

def _make_smpl_joints():
    """SMPL 24-joint 3D positions for a standing figure."""
    joints = [[0.0, 0.0, 0.0]] * 24
    joints[0] = [0.0, 0.9, 0.0]     # pelvis
    joints[1] = [-0.1, 0.9, 0.0]    # left hip
    joints[2] = [0.1, 0.9, 0.0]     # right hip
    joints[3] = [0.0, 1.05, 0.0]    # spine1
    joints[4] = [-0.1, 0.5, 0.0]    # left knee
    joints[5] = [0.1, 0.5, 0.0]     # right knee
    joints[6] = [0.0, 1.2, 0.0]     # spine2
    joints[7] = [-0.1, 0.05, 0.0]   # left ankle
    joints[8] = [0.1, 0.05, 0.0]    # right ankle
    joints[9] = [0.0, 1.35, 0.0]    # spine3
    joints[12] = [0.0, 1.5, 0.0]    # neck
    joints[15] = [0.0, 1.6, 0.0]    # head
    joints[16] = [-0.2, 1.4, 0.0]   # left shoulder
    joints[17] = [0.2, 1.4, 0.0]    # right shoulder
    joints[18] = [-0.4, 1.2, 0.0]   # left elbow
    joints[19] = [0.4, 1.2, 0.0]    # right elbow
    joints[20] = [-0.5, 1.0, 0.0]   # left wrist
    joints[21] = [0.5, 1.0, 0.0]    # right wrist
    return joints


def _make_2d_pose():
    """RTMPose-style 2D pose dict with named keypoints."""
    return {
        "left_shoulder": {"x": 0.4, "y": 0.3, "confidence": 0.9},
        "right_shoulder": {"x": 0.6, "y": 0.3, "confidence": 0.9},
        "left_hip": {"x": 0.45, "y": 0.5, "confidence": 0.9},
        "right_hip": {"x": 0.55, "y": 0.5, "confidence": 0.9},
        "left_knee": {"x": 0.45, "y": 0.65, "confidence": 0.9},
        "right_knee": {"x": 0.55, "y": 0.65, "confidence": 0.9},
        "left_ankle": {"x": 0.45, "y": 0.8, "confidence": 0.9},
        "right_ankle": {"x": 0.55, "y": 0.8, "confidence": 0.9},
        "left_elbow": {"x": 0.35, "y": 0.4, "confidence": 0.9},
        "right_elbow": {"x": 0.65, "y": 0.4, "confidence": 0.9},
        "left_wrist": {"x": 0.3, "y": 0.5, "confidence": 0.9},
        "right_wrist": {"x": 0.7, "y": 0.5, "confidence": 0.9},
        "left_ear": {"x": 0.47, "y": 0.15, "confidence": 0.9},
        "right_ear": {"x": 0.53, "y": 0.15, "confidence": 0.9},
    }


# ---------------------------------------------------------------------------
# 3D biometric extraction
# ---------------------------------------------------------------------------

class TestExtract3D:
    def test_valid_joints(self):
        sig = extract_biometric_signature_3d(_make_smpl_joints())
        assert sig is not None
        assert sig.available_count == 6
        assert sig.shoulder_hip_ratio is not None
        assert sig.torso_leg_ratio is not None
        assert sig.upper_lower_arm_ratio is not None
        assert sig.thigh_shin_ratio is not None
        assert sig.head_shoulder_ratio is not None
        assert sig.arm_body_ratio is not None

    def test_none_input(self):
        assert extract_biometric_signature_3d(None) is None

    def test_empty_input(self):
        assert extract_biometric_signature_3d([]) is None

    def test_too_few_joints(self):
        assert extract_biometric_signature_3d([[0, 0, 0]] * 10) is None

    def test_malformed_joints(self):
        joints = [[0, 0]] * 24  # 2D instead of 3D
        assert extract_biometric_signature_3d(joints) is None

    def test_ratios_are_scale_invariant(self):
        joints_1x = _make_smpl_joints()
        joints_2x = [[c * 2 for c in j] for j in joints_1x]
        sig_1x = extract_biometric_signature_3d(joints_1x)
        sig_2x = extract_biometric_signature_3d(joints_2x)
        assert sig_1x is not None and sig_2x is not None
        assert abs(sig_1x.shoulder_hip_ratio - sig_2x.shoulder_hip_ratio) < 1e-6
        assert abs(sig_1x.torso_leg_ratio - sig_2x.torso_leg_ratio) < 1e-6
        assert abs(sig_1x.thigh_shin_ratio - sig_2x.thigh_shin_ratio) < 1e-6

    def test_3d_uses_depth(self):
        """3D ratios should differ from 2D when depth varies asymmetrically."""
        joints = _make_smpl_joints()
        sig_flat = extract_biometric_signature_3d(joints)
        # Move left thigh segment forward in Z but not shin — changes thigh/shin ratio
        joints_depth = [list(j) for j in joints]
        joints_depth[1][2] = 0.4   # left hip forward
        joints_depth[4][2] = 0.2   # left knee partially forward
        # ankle stays at z=0 — shin is now longer in 3D than thigh
        sig_depth = extract_biometric_signature_3d(joints_depth)
        assert sig_flat is not None and sig_depth is not None
        assert sig_flat.thigh_shin_ratio != sig_depth.thigh_shin_ratio

    def test_returns_biometric_signature_type(self):
        sig = extract_biometric_signature_3d(_make_smpl_joints())
        assert isinstance(sig, BiometricSignature)


# ---------------------------------------------------------------------------
# 2D biometric extraction (existing, regression)
# ---------------------------------------------------------------------------

class TestExtract2D:
    def test_valid_pose(self):
        sig = extract_biometric_signature(_make_2d_pose())
        assert sig is not None
        assert sig.available_count >= 5

    def test_empty_dict(self):
        assert extract_biometric_signature({}) is None

    def test_low_confidence_skipped(self):
        pose = _make_2d_pose()
        for key in pose:
            pose[key]["confidence"] = 0.1
        assert extract_biometric_signature(pose) is None


# ---------------------------------------------------------------------------
# Similarity and merging (regression)
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical_signatures(self):
        sig = extract_biometric_signature_3d(_make_smpl_joints())
        assert signature_similarity(sig, sig) == 1.0

    def test_different_signatures(self):
        joints_a = _make_smpl_joints()
        joints_b = _make_smpl_joints()
        joints_b[16] = [-0.3, 1.4, 0.0]  # wider shoulders
        joints_b[17] = [0.3, 1.4, 0.0]
        sig_a = extract_biometric_signature_3d(joints_a)
        sig_b = extract_biometric_signature_3d(joints_b)
        sim = signature_similarity(sig_a, sig_b)
        assert 0.0 < sim < 1.0

    def test_none_returns_neutral(self):
        assert signature_similarity(None, None) == 0.5
        sig = extract_biometric_signature_3d(_make_smpl_joints())
        assert signature_similarity(sig, None) == 0.5

    def test_merge_ema(self):
        sig_a = extract_biometric_signature_3d(_make_smpl_joints())
        joints_b = _make_smpl_joints()
        joints_b[16] = [-0.3, 1.4, 0.0]
        joints_b[17] = [0.3, 1.4, 0.0]
        sig_b = extract_biometric_signature_3d(joints_b)
        merged = merge_signatures(sig_a, sig_b, alpha=0.5)
        # Merged shoulder_hip_ratio should be between the two
        assert sig_a.shoulder_hip_ratio < merged.shoulder_hip_ratio < sig_b.shoulder_hip_ratio

"""IoU + centroid tracker with ByteTrack core, identity overlay (biometric, Re-ID,
appearance, motion), and graveyard re-identification for robust occlusion recovery.

The ByteTrack core handles the MOT problem (which bbox matches which bbox).
The identity layer handles "which bbox is which dancer" via biometric signatures,
Re-ID embeddings, appearance histograms, and motion coherence.
"""
import logging

import numpy as np

from app.pipeline.biometrics import (
    BiometricSignature, extract_biometric_signature, merge_signatures, signature_similarity,
)
from app.pipeline.bytetrack import BYTETracker, STrack, TrackState
from app.pipeline.reid import cosine_similarity as reid_cosine_similarity, merge_embeddings
from app.pipeline.pose_config import (
    TRACKER_RESEED_IDENTITY_THRESHOLD,
    TRACKER_COHERENCE_THRESHOLD,
    TRACKER_FORMATION_RATIO_LIMIT,
    TRACKER_REID_BASE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def compute_bbox(keypoints: np.ndarray, scores: np.ndarray, min_conf: float = 0.3) -> tuple[float, float, float, float] | None:
    """Compute bounding box from valid keypoints. Returns (x_min, y_min, x_max, y_max) or None."""
    valid = scores > min_conf
    if valid.sum() < 3:
        return None
    pts = keypoints[valid]
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute intersection-over-union between two (x_min, y_min, x_max, y_max) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _bbox_centroid(bbox: tuple) -> tuple[float, float]:
    """Return (cx, cy) centroid of a bbox."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def centroid_distance(box_a: tuple, box_b: tuple) -> float:
    """Euclidean distance between centroids of two boxes."""
    ca = _bbox_centroid(box_a)
    cb = _bbox_centroid(box_b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def _bbox_area(bbox: tuple) -> float:
    """Compute area of a bounding box."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


class SimpleTracker:
    """Track persons across frames using ByteTrack core with identity overlay.

    ByteTrack handles frame-to-frame bbox association via Kalman filter + IoU.
    The identity layer maps ByteTrack's internal IDs to stable dancer IDs
    using appearance, biometric signatures, Re-ID embeddings, and motion coherence.

    Occlusion handling:
    1. ByteTrack Kalman filter — predicts position during occlusion
    2. Two-pass matching — high-conf then low-conf detections
    3. Appearance matching — color histograms for identity disambiguation
    4. Graveyard re-ID — expired tracks recovered via identity signals
    5. Group coherence — formation distance constraints for dancer groups
    """

    def __init__(
        self,
        iou_threshold: float = 0.1,
        max_centroid_dist: float = 0.3,
        max_missing: int = 90,
        max_size_ratio: float = 2.5,
        graveyard_frames: int = 300,
        appearance_weight: float = 0.3,
        velocity_smoothing: float = 0.3,
        effective_fps: float = 30.0,
        relock_min_wait: int = 15,
    ):
        # Scale frame-based thresholds by FPS
        fps_scale = effective_fps / 30.0
        self.iou_threshold = iou_threshold
        self.max_centroid_dist = max_centroid_dist
        self.max_missing = int(max_missing * fps_scale)
        self.max_size_ratio = max_size_ratio
        self.graveyard_frames = int(graveyard_frames * fps_scale)
        self.appearance_weight = appearance_weight
        self.velocity_smoothing = velocity_smoothing
        self.effective_fps = effective_fps
        self.relock_min_wait = int(relock_min_wait * fps_scale)
        self.reseed_identity_threshold = TRACKER_RESEED_IDENTITY_THRESHOLD
        self.coherence_threshold = TRACKER_COHERENCE_THRESHOLD
        self.formation_ratio_limit = TRACKER_FORMATION_RATIO_LIMIT
        self.reid_base_threshold = TRACKER_REID_BASE_THRESHOLD

        # ByteTrack core — handles the actual MOT matching
        # Use low track_thresh since RTMPose detections don't have real confidence.
        # match_thresh=0.8 means IoU cost < 0.8 → IoU > 0.2 to match.
        # second_match_thresh=0.9 means IoU > 0.1 for low-conf second pass.
        self._bytetrack = BYTETracker(
            track_thresh=0.5,
            match_thresh=0.8,
            low_thresh=0.1,
            second_match_thresh=0.9,
            max_time_lost=self.max_missing,
            min_hits=1,
        )

        # ID mapping: bytetrack_id -> dancer_id (our stable IDs)
        self._bt_to_dancer: dict[int, int] = {}
        # Reverse: dancer_id -> bytetrack_id
        self._dancer_to_bt: dict[int, int] = {}

        self.next_id = 0
        self.active_tracks: dict[int, tuple] = {}  # dancer_id -> last bbox
        self.missing_count: dict[int, int] = {}
        self.last_seen_bbox: dict[int, tuple] = {}
        self.track_avg_area: dict[int, float] = {}
        self.track_velocity: dict[int, tuple[float, float]] = {}
        self.track_histogram: dict[int, list[float]] = {}

        # Graveyard: tracks that exceeded max_missing but can be re-identified
        self.graveyard: dict[int, tuple] = {}
        self.graveyard_area: dict[int, float] = {}
        self.graveyard_age: dict[int, int] = {}
        self.graveyard_histogram: dict[int, list[float]] = {}
        self.graveyard_velocity: dict[int, tuple[float, float]] = {}

        # Group tracking
        self.group_ids: set[int] = set()
        self.group_distances: dict[tuple[int, int], float] = {}

        # Identity signals
        self.track_biometric: dict[int, BiometricSignature] = {}
        self.graveyard_biometric: dict[int, BiometricSignature] = {}
        self.track_embedding: dict[int, np.ndarray] = {}
        self.graveyard_embedding: dict[int, np.ndarray] = {}
        self.track_motion: dict[int, np.ndarray] = {}
        self._predicted_group_motion: np.ndarray | None = None

        # Re-ID state
        self.relock_candidates: dict[int, dict] = {}
        self.graveyard_candidates: dict[int, dict] = {}
        self.relock_confirm = 3
        self.reid_fail_count: dict[int, int] = {}
        self.reid_frame: dict[int, int] = {}

        self._reseed_pending = False
        self._frame_counter = 0

    def seed(self, known_bboxes: dict[int, tuple], histograms: dict[int, list[float]] | None = None,
             group_ids: set[int] | None = None, biometrics: dict[int, BiometricSignature] | None = None,
             embeddings: dict[int, np.ndarray] | None = None):
        """Seed the tracker with known track_id -> bbox mappings from a prior detection pass."""
        self._original_bboxes = dict(known_bboxes)
        self._original_histograms = dict(histograms) if histograms else {}
        self._original_group_ids = set(group_ids) if group_ids else set(known_bboxes.keys())

        # Set STrack ID counter high enough to not collide with dancer IDs
        max_dancer_id = max(known_bboxes.keys()) if known_bboxes else -1
        STrack.set_next_id(max_dancer_id + 1000)

        for tid, bbox in known_bboxes.items():
            # Add track directly to ByteTrack with a specific ID
            self._bytetrack.add_track(bbox, track_id=tid, score=1.0)
            # For seeded tracks, bytetrack ID == dancer ID
            self._bt_to_dancer[tid] = tid
            self._dancer_to_bt[tid] = tid

            self.active_tracks[tid] = bbox
            self.missing_count[tid] = 0
            self.last_seen_bbox[tid] = bbox
            self.track_avg_area[tid] = _bbox_area(bbox)
            self.track_velocity[tid] = (0.0, 0.0)

        if histograms:
            for tid, hist in histograms.items():
                self.track_histogram[tid] = hist
        if biometrics:
            for tid, bio in biometrics.items():
                if bio is not None:
                    self.track_biometric[tid] = bio
        if embeddings:
            for tid, emb in embeddings.items():
                if emb is not None:
                    self.track_embedding[tid] = emb

        if group_ids:
            self.group_ids = set(group_ids)
        else:
            self.group_ids = set(known_bboxes.keys())

        self._update_group_distances()
        self.next_id = max_dancer_id + 1 if known_bboxes else 0
        self._first_update_after_seed = True

    def reseed(self):
        """Set reseed-pending flag for identity-based re-matching on the next update().

        Moves all group members to graveyard so they can be identity-matched fresh.
        """
        if not hasattr(self, '_original_bboxes') or not self._original_bboxes:
            return

        for tid in list(self.active_tracks.keys()):
            if tid in self._original_group_ids:
                seen = self.last_seen_bbox.get(tid, self.active_tracks[tid])
                self._move_to_graveyard(tid, seen)

        for tid in self._original_group_ids:
            if tid in self.graveyard:
                self.graveyard_age[tid] = max(self.graveyard_age.get(tid, 0), self.relock_min_wait)

        # Reset ByteTrack — remove all tracked stracks for group members
        self._bytetrack.tracked_stracks = [
            s for s in self._bytetrack.tracked_stracks
            if self._bt_to_dancer.get(s.track_id) not in self._original_group_ids
        ]
        self._bytetrack.lost_stracks = [
            s for s in self._bytetrack.lost_stracks
            if self._bt_to_dancer.get(s.track_id) not in self._original_group_ids
        ]

        self._reseed_pending = True
        self._reseed_grace_frames = int(10 * (self.effective_fps / 30.0))
        logger.info(f"Reseed pending — identity-based re-matching will run on next update "
                    f"for tracks {sorted(self._original_group_ids)}")

    def _update_group_distances(self):
        """Update smoothed pairwise centroid distances between group members."""
        gids = [tid for tid in self.group_ids
                if tid in self.active_tracks and self.missing_count.get(tid, 0) == 0]
        alpha = 0.3
        for i, tid_a in enumerate(gids):
            for tid_b in gids[i + 1:]:
                pair = (min(tid_a, tid_b), max(tid_a, tid_b))
                dist = centroid_distance(self.active_tracks[tid_a], self.active_tracks[tid_b])
                prev = self.group_distances.get(pair)
                if prev is None:
                    self.group_distances[pair] = dist
                else:
                    self.group_distances[pair] = prev * (1 - alpha) + dist * alpha

    def _group_coherence_score(self, candidate_bbox: tuple, candidate_tid: int,
                                proposed_assignments: dict[int, tuple]) -> float:
        """Score how well a candidate assignment preserves the group's spatial formation."""
        if candidate_tid not in self.group_ids or len(self.group_distances) == 0:
            return 1.0

        penalties = []
        for other_tid in self.group_ids:
            if other_tid == candidate_tid:
                continue
            pair = (min(candidate_tid, other_tid), max(candidate_tid, other_tid))
            expected_dist = self.group_distances.get(pair)
            if expected_dist is None or expected_dist <= 0:
                continue
            other_bbox = proposed_assignments.get(other_tid)
            if other_bbox is None:
                other_bbox = self.active_tracks.get(other_tid)
            if other_bbox is None:
                continue
            actual_dist = centroid_distance(candidate_bbox, other_bbox)
            ratio = actual_dist / expected_dist if expected_dist > 0 else 1.0
            if ratio > self.formation_ratio_limit or ratio < 1.0 / self.formation_ratio_limit:
                penalties.append(abs(1.0 - ratio))

        if not penalties:
            return 1.0
        avg_penalty = sum(penalties) / len(penalties)
        return max(0.0, 1.0 - avg_penalty)

    def _size_compatible(self, bbox: tuple, track_id: int) -> bool:
        """Check if a detection's size is compatible with a track's historical size."""
        avg_area = self.track_avg_area.get(track_id)
        if avg_area is None or avg_area <= 0:
            return True
        det_area = _bbox_area(bbox)
        if det_area <= 0:
            return False
        ratio = det_area / avg_area if det_area >= avg_area else avg_area / det_area
        return ratio <= self.max_size_ratio

    def _appearance_similarity(self, hist: list[float] | None, track_id: int) -> float:
        """Return histogram intersection similarity (0-1) or 0.5 if no histogram available."""
        if not hist:
            return 0.5
        track_hist = self.track_histogram.get(track_id)
        if not track_hist or len(track_hist) != len(hist):
            return 0.5
        return float(sum(min(a, b) for a, b in zip(hist, track_hist)))

    def _update_velocity(self, track_id: int, old_bbox: tuple, new_bbox: tuple):
        """Update smoothed velocity for a track."""
        old_c = _bbox_centroid(old_bbox)
        new_c = _bbox_centroid(new_bbox)
        dx = new_c[0] - old_c[0]
        dy = new_c[1] - old_c[1]
        alpha = self.velocity_smoothing
        prev = self.track_velocity.get(track_id, (0.0, 0.0))
        self.track_velocity[track_id] = (
            prev[0] * (1 - alpha) + dx * alpha,
            prev[1] * (1 - alpha) + dy * alpha,
        )

    def _update_avg_area(self, track_id: int, bbox: tuple, alpha: float = 0.1):
        """Update running average area for a track."""
        area = _bbox_area(bbox)
        if area <= 0:
            return
        prev = self.track_avg_area.get(track_id)
        if prev is None or prev <= 0:
            self.track_avg_area[track_id] = area
        else:
            self.track_avg_area[track_id] = prev * (1 - alpha) + area * alpha

    def _update_histogram(self, track_id: int, hist: list[float] | None, alpha: float = 0.1):
        """Update running average color histogram for a track."""
        if not hist:
            return
        prev = self.track_histogram.get(track_id)
        if not prev or len(prev) != len(hist):
            self.track_histogram[track_id] = list(hist)
        else:
            self.track_histogram[track_id] = [
                p * (1 - alpha) + h * alpha for p, h in zip(prev, hist)
            ]

    def _move_to_graveyard(self, tid: int, seen_bbox: tuple):
        """Move a track to the graveyard, preserving all identity signals."""
        self.graveyard[tid] = seen_bbox
        self.graveyard_area[tid] = self.track_avg_area.get(tid, 0)
        self.graveyard_age[tid] = 0
        self.graveyard_velocity[tid] = self.track_velocity.get(tid, (0.0, 0.0))
        if tid in self.track_histogram:
            self.graveyard_histogram[tid] = self.track_histogram[tid]
        if tid in self.track_biometric:
            self.graveyard_biometric[tid] = self.track_biometric[tid]
        if tid in self.track_embedding:
            self.graveyard_embedding[tid] = self.track_embedding[tid]
        # Clean from active
        self.active_tracks.pop(tid, None)
        self.missing_count.pop(tid, None)
        self.graveyard_candidates.pop(tid, None)
        self.reid_fail_count.pop(tid, None)
        reid_at = self.reid_frame.pop(tid, None)
        if reid_at is not None:
            frames_since = self._frame_counter - reid_at
            if frames_since < self.max_missing * 2:
                self.reid_fail_count[tid] = self.reid_fail_count.get(tid, 0) + 1
        if tid in self.group_ids:
            logger.warning(f"GROUP TRACK {tid} moved to graveyard "
                           f"(last seen at {_bbox_centroid(seen_bbox)})")

    def _clear_graveyard_entry(self, g_tid: int):
        """Remove a graveyard entry and all associated identity data."""
        if g_tid in self.group_ids:
            logger.warning(f"GROUP TRACK {g_tid} EXPIRED from graveyard at frame {self._frame_counter}")
        self.graveyard.pop(g_tid, None)
        self.graveyard_area.pop(g_tid, None)
        self.graveyard_age.pop(g_tid, None)
        self.graveyard_velocity.pop(g_tid, None)
        self.graveyard_histogram.pop(g_tid, None)
        self.graveyard_biometric.pop(g_tid, None)
        self.graveyard_embedding.pop(g_tid, None)
        self.reid_fail_count.pop(g_tid, None)

    def _update_biometric(self, track_id: int, bio: BiometricSignature | None, alpha: float = 0.1):
        if bio is None:
            return
        prev = self.track_biometric.get(track_id)
        if prev is None:
            self.track_biometric[track_id] = bio
        else:
            self.track_biometric[track_id] = merge_signatures(prev, bio, alpha)

    def _update_embedding(self, track_id: int, emb: np.ndarray | None, alpha: float = 0.1):
        if emb is None:
            return
        prev = self.track_embedding.get(track_id)
        if prev is None:
            self.track_embedding[track_id] = emb
        else:
            self.track_embedding[track_id] = merge_embeddings(prev, emb, alpha)

    def _update_motion(self, track_id: int, motion: np.ndarray | None, alpha: float = 0.3):
        if motion is None:
            return
        prev = self.track_motion.get(track_id)
        if prev is None:
            self.track_motion[track_id] = motion.copy()
        else:
            self.track_motion[track_id] = prev * (1 - alpha) + motion * alpha

    def _update_predicted_group_motion(self):
        """Compute average motion state from visible group members."""
        visible = []
        for tid in self.group_ids:
            if (tid in self.track_motion
                    and self.missing_count.get(tid, 0) == 0
                    and tid in self.active_tracks):
                visible.append(self.track_motion[tid])
        if len(visible) >= 1:
            self._predicted_group_motion = np.mean(visible, axis=0)
        else:
            self._predicted_group_motion = None

    def _motion_coherence(self, det_motion: np.ndarray | None) -> float:
        if det_motion is None or self._predicted_group_motion is None:
            return 0.5
        a = det_motion
        b = self._predicted_group_motion
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.5
        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        return (cos_sim + 1.0) / 2.0

    def _identity_score(self, det_hist, det_bio, det_emb, track_id: int,
                        use_graveyard: bool = False,
                        det_motion: np.ndarray | None = None) -> float:
        """Compute blended identity score using histogram + biometric + Re-ID + motion."""
        scores = {}
        weights = {}

        # Histogram
        if use_graveyard:
            track_hist = self.graveyard_histogram.get(track_id)
        else:
            track_hist = self.track_histogram.get(track_id)
        if det_hist and track_hist and len(det_hist) == len(track_hist):
            scores["hist"] = float(sum(min(a, b) for a, b in zip(det_hist, track_hist)))
            weights["hist"] = 0.20
        else:
            scores["hist"] = 0.5
            weights["hist"] = 0.08

        # Biometric
        if use_graveyard:
            track_bio = self.graveyard_biometric.get(track_id)
        else:
            track_bio = self.track_biometric.get(track_id)
        bio_sim = signature_similarity(det_bio, track_bio)
        if bio_sim != 0.5 or (det_bio is not None and track_bio is not None):
            scores["bio"] = bio_sim
            weights["bio"] = 0.20
        else:
            scores["bio"] = 0.5
            weights["bio"] = 0.05

        # Re-ID embedding
        if use_graveyard:
            track_emb = self.graveyard_embedding.get(track_id)
        else:
            track_emb = self.track_embedding.get(track_id)
        reid_sim = reid_cosine_similarity(det_emb, track_emb)
        if reid_sim != 0.5 or (det_emb is not None and track_emb is not None):
            scores["reid"] = reid_sim
            weights["reid"] = 0.25
        else:
            scores["reid"] = 0.5
            weights["reid"] = 0.05

        # Motion coherence
        motion_sim = self._motion_coherence(det_motion)
        if motion_sim != 0.5 and det_motion is not None and self._predicted_group_motion is not None:
            scores["motion"] = motion_sim
            weights["motion"] = 0.15
        else:
            scores["motion"] = 0.5
            weights["motion"] = 0.02

        total_w = sum(weights.values())
        if total_w < 1e-6:
            return 0.5
        return sum(scores[k] * weights[k] for k in scores) / total_w

    def _recover_from_graveyard(self, tid: int, bbox: tuple, hist=None, bio=None, emb=None):
        """Move a track from graveyard back to active state."""
        self.active_tracks[tid] = bbox
        self.missing_count[tid] = 0
        self.last_seen_bbox[tid] = bbox
        self._update_avg_area(tid, bbox)
        self.track_velocity[tid] = (0.0, 0.0)
        if tid in self.graveyard_histogram:
            self.track_histogram[tid] = self.graveyard_histogram.pop(tid)
        if tid in self.graveyard_biometric:
            self.track_biometric[tid] = self.graveyard_biometric.pop(tid)
        if tid in self.graveyard_embedding:
            self.track_embedding[tid] = self.graveyard_embedding.pop(tid)
        self.graveyard.pop(tid, None)
        self.graveyard_area.pop(tid, None)
        self.graveyard_age.pop(tid, None)
        self.graveyard_velocity.pop(tid, None)
        self.graveyard_candidates.pop(tid, None)
        self._update_histogram(tid, hist)
        self._update_biometric(tid, bio)
        self._update_embedding(tid, emb)
        self.reid_frame[tid] = self._frame_counter

        # Re-add to ByteTrack as a tracked strack
        # Use dancer ID directly since we map it
        bt_id = tid  # for recovered group tracks, keep same ID
        strack = STrack.__new__(STrack)
        from app.pipeline.bytetrack import KalmanFilter, _bbox_to_cxywh
        strack.kf = KalmanFilter()
        strack.kf.init(_bbox_to_cxywh(bbox))
        strack.track_id = bt_id
        strack.score = 1.0
        strack.state = TrackState.Tracked
        strack.is_activated = True
        strack.frame_id = self._bytetrack.frame_id
        strack.start_frame = self._bytetrack.frame_id
        strack.time_since_update = 0
        strack._bbox = bbox
        self._bytetrack.tracked_stracks.append(strack)
        self._bt_to_dancer[bt_id] = tid
        self._dancer_to_bt[tid] = bt_id

    def _compute_detection_scores(self, bboxes: list[tuple], histograms, biometrics, embeddings) -> list[float]:
        """Compute confidence scores for detections.

        For seeded trackers, detections near known group members get high confidence.
        This enables ByteTrack's two-pass matching to work: dancers get matched first
        (high-conf pass), transients get matched second (low-conf pass).
        """
        if not self.group_ids or not self.active_tracks:
            return [1.0] * len(bboxes)

        scores = []
        for i, bbox in enumerate(bboxes):
            # Base score: proximity to any active group member
            best_iou = 0.0
            for tid in self.group_ids:
                t_bbox = self.active_tracks.get(tid)
                if t_bbox is None:
                    continue
                iou = compute_iou(bbox, t_bbox)
                if iou > best_iou:
                    best_iou = iou

            # Also check centroid distance
            best_dist = 999.0
            for tid in self.group_ids:
                t_bbox = self.active_tracks.get(tid)
                if t_bbox is None:
                    continue
                d = centroid_distance(bbox, t_bbox)
                if d < best_dist:
                    best_dist = d

            # High confidence if close to a group member
            if best_iou > 0.3 or best_dist < 0.15:
                scores.append(1.0)
            elif best_iou > 0.1 or best_dist < 0.3:
                scores.append(0.7)
            else:
                scores.append(0.3)  # low-conf → second pass

        return scores

    def update(self, bboxes: list[tuple], histograms: list[list[float]] | None = None,
               biometrics: list[BiometricSignature | None] | None = None,
               embeddings: list[np.ndarray | None] | None = None,
               motions: list[np.ndarray | None] | None = None) -> list[int]:
        """Assign track IDs to a list of bounding boxes for one frame.

        Returns a list of track_ids parallel to the input bboxes.
        """
        hists = histograms or [None] * len(bboxes)
        bios = biometrics or [None] * len(bboxes)
        embs = embeddings or [None] * len(bboxes)
        mots = motions or [None] * len(bboxes)

        # --- Reseed handling: identity-based re-matching ---
        if self._reseed_pending and bboxes:
            self._reseed_pending = False
            result = self._handle_reseed(bboxes, hists, bios, embs, mots)
            if result is not None:
                return result

        if not bboxes:
            # No detections — let ByteTrack predict and age out tracks
            self._bytetrack.update([], [])
            self._sync_from_bytetrack()
            self._age_graveyard()
            self._frame_counter += 1
            return []

        if not self.active_tracks and not self._bytetrack.tracked_stracks:
            # First frame: assign new IDs via ByteTrack
            scores = [1.0] * len(bboxes)
            bt_results = self._bytetrack.update(bboxes, scores)

            assignments = [None] * len(bboxes)
            for det_idx, bt_id in bt_results:
                dancer_id = self.next_id
                self.next_id += 1
                self._bt_to_dancer[bt_id] = dancer_id
                self._dancer_to_bt[dancer_id] = bt_id
                assignments[det_idx] = dancer_id
                self.active_tracks[dancer_id] = bboxes[det_idx]
                self.missing_count[dancer_id] = 0
                self.last_seen_bbox[dancer_id] = bboxes[det_idx]
                self.track_avg_area[dancer_id] = _bbox_area(bboxes[det_idx])
                self.track_velocity[dancer_id] = (0.0, 0.0)
                self._update_histogram(dancer_id, hists[det_idx])

            # Fill any Nones with new IDs
            for i in range(len(assignments)):
                if assignments[i] is None:
                    assignments[i] = self.next_id
                    self.next_id += 1
            self._frame_counter += 1
            return assignments

        # --- Main tracking path: delegate to ByteTrack ---
        det_scores = self._compute_detection_scores(bboxes, hists, bios, embs)
        bt_results = self._bytetrack.update(bboxes, det_scores)

        # Map ByteTrack results to dancer IDs
        assignments = [None] * len(bboxes)
        matched_dancer_ids = set()
        bt_matched_dets = set()

        for det_idx, bt_id in bt_results:
            dancer_id = self._bt_to_dancer.get(bt_id)
            if dancer_id is None:
                # New track from ByteTrack — assign new dancer ID
                # But cap transient tracks when group tracking is active
                if self.group_ids:
                    max_non_group = max(10, len(self.group_ids) * 3)
                    current_non_group = sum(1 for t in self.active_tracks if t not in self.group_ids)
                    if current_non_group >= max_non_group:
                        assignments[det_idx] = -1
                        continue

                dancer_id = self.next_id
                self.next_id += 1
                self._bt_to_dancer[bt_id] = dancer_id
                self._dancer_to_bt[dancer_id] = bt_id

            assignments[det_idx] = dancer_id
            matched_dancer_ids.add(dancer_id)
            bt_matched_dets.add(det_idx)

        # --- Group member rescue: steal detections from transient tracks ---
        # ByteTrack may have matched a detection to a new transient track when it
        # actually belongs to a missing group member. Check by IoU + proximity.
        if self.group_ids:
            missing_group = [
                tid for tid in self.group_ids
                if tid in self.active_tracks
                and tid not in matched_dancer_ids
                and self.missing_count.get(tid, 0) < self.max_missing
            ]
            # Also include group members in the graveyard
            missing_group += [
                tid for tid in self.group_ids
                if tid in self.graveyard
                and tid not in matched_dancer_ids
            ]
            if missing_group:
                # Find detections assigned to non-group (transient) tracks
                stealable = [
                    (det_idx, assignments[det_idx])
                    for det_idx in range(len(bboxes))
                    if assignments[det_idx] is not None
                    and assignments[det_idx] != -1
                    and assignments[det_idx] not in self.group_ids
                ]
                for g_tid in missing_group:
                    # Get the expected position for this group member
                    expected_bbox = self.active_tracks.get(g_tid)
                    if expected_bbox is None:
                        expected_bbox = self.graveyard.get(g_tid)
                    if expected_bbox is None:
                        expected_bbox = self._original_bboxes.get(g_tid)
                    if expected_bbox is None:
                        continue

                    best_det = -1
                    best_iou = 0.15  # minimum IoU to steal
                    best_dist = self.max_centroid_dist * 2.0

                    for det_idx, transient_id in stealable:
                        iou = compute_iou(bboxes[det_idx], expected_bbox)
                        dist = centroid_distance(bboxes[det_idx], expected_bbox)
                        # Prefer IoU, fall back to centroid
                        if iou > best_iou:
                            best_iou = iou
                            best_det = det_idx
                        elif iou < 0.01 and dist < best_dist:
                            best_dist = dist
                            best_det = det_idx

                    if best_det >= 0:
                        old_id = assignments[best_det]
                        assignments[best_det] = g_tid
                        matched_dancer_ids.add(g_tid)
                        matched_dancer_ids.discard(old_id)
                        # Remove the old stealable entry
                        stealable = [(d, t) for d, t in stealable if d != best_det]
                        # If group member was in graveyard, recover it
                        if g_tid in self.graveyard:
                            self._recover_from_graveyard(
                                g_tid, bboxes[best_det],
                                hists[best_det] if hists else None,
                                bios[best_det] if bios else None,
                                embs[best_det] if embs else None)
                            logger.info(f"Rescued group track {g_tid} from graveyard "
                                        f"(stole from transient {old_id}, "
                                        f"iou={best_iou:.3f})")
                        else:
                            logger.debug(f"Rescued group track {g_tid} from transient {old_id}")

        # --- Graveyard re-ID: unmatched detections vs graveyarded tracks ---
        unmatched_dets = [i for i in range(len(bboxes)) if assignments[i] is None]
        recovered_dets = set()

        if unmatched_dets and self.graveyard:
            for det_idx in unmatched_dets:
                g_tid = self._try_graveyard_match(
                    det_idx, bboxes[det_idx], hists[det_idx], bios[det_idx],
                    embs[det_idx], mots[det_idx], assignments, bboxes)
                if g_tid is not None:
                    assignments[det_idx] = g_tid
                    matched_dancer_ids.add(g_tid)
                    recovered_dets.add(det_idx)

        # Identity re-lock for group members still in graveyard
        still_unmatched = [i for i in unmatched_dets if i not in recovered_dets]
        if still_unmatched:
            unrecovered_group = [
                g_tid for g_tid in self.graveyard
                if g_tid in self.group_ids
                and self.graveyard_age.get(g_tid, 0) >= self.relock_min_wait
                and g_tid not in matched_dancer_ids
            ]
            if unrecovered_group:
                for det_idx in still_unmatched:
                    g_tid = self._try_identity_relock(
                        det_idx, bboxes[det_idx], hists[det_idx], bios[det_idx],
                        embs[det_idx], mots[det_idx], unrecovered_group)
                    if g_tid is not None:
                        assignments[det_idx] = g_tid
                        matched_dancer_ids.add(g_tid)
                        unrecovered_group.remove(g_tid)

        # Assign -1 to remaining unmatched detections (suppressed)
        for i in range(len(assignments)):
            if assignments[i] is None:
                if self.group_ids:
                    max_non_group = max(10, len(self.group_ids) * 3)
                    current_non_group = sum(1 for t in self.active_tracks if t not in self.group_ids)
                    if current_non_group >= max_non_group:
                        assignments[i] = -1
                        continue
                # New track
                new_id = self.next_id
                self.next_id += 1
                assignments[i] = new_id

        # --- Update state for all assigned detections ---
        new_active = {}
        new_missing = {}

        for det_idx, dancer_id in enumerate(assignments):
            if dancer_id == -1:
                continue
            old_bbox = self.active_tracks.get(dancer_id)
            new_active[dancer_id] = bboxes[det_idx]
            new_missing[dancer_id] = 0
            self.last_seen_bbox[dancer_id] = bboxes[det_idx]
            self._update_avg_area(dancer_id, bboxes[det_idx])
            self._update_histogram(dancer_id, hists[det_idx])
            self._update_biometric(dancer_id, bios[det_idx])
            self._update_embedding(dancer_id, embs[det_idx])
            self._update_motion(dancer_id, mots[det_idx])
            if old_bbox is not None:
                self._update_velocity(dancer_id, old_bbox, bboxes[det_idx])
            elif dancer_id not in self.track_velocity:
                self.track_velocity[dancer_id] = (0.0, 0.0)

        # Keep unmatched active tracks alive with predicted positions from ByteTrack
        for dancer_id, bbox in list(self.active_tracks.items()):
            if dancer_id not in new_active:
                age = self.missing_count.get(dancer_id, 0) + 1
                if age <= self.max_missing:
                    # Use ByteTrack's Kalman prediction if available
                    bt_id = self._dancer_to_bt.get(dancer_id)
                    pred_bbox = bbox
                    if bt_id is not None:
                        for s in self._bytetrack.tracked_stracks + self._bytetrack.lost_stracks:
                            if s.track_id == bt_id:
                                pred_bbox = s.bbox
                                break
                    new_active[dancer_id] = pred_bbox
                    new_missing[dancer_id] = age
                else:
                    seen = self.last_seen_bbox.get(dancer_id, bbox)
                    self._move_to_graveyard(dancer_id, seen)

        self.active_tracks = new_active
        self.missing_count = new_missing

        # Age graveyard
        self._age_graveyard()

        self._frame_counter += 1

        # Clean stale candidates
        stale = [tid for tid, c in self.relock_candidates.items()
                 if self._frame_counter - c["frame_idx"] > 5]
        for tid in stale:
            del self.relock_candidates[tid]
        stale_gc = [tid for tid, c in self.graveyard_candidates.items()
                    if self._frame_counter - c["frame_idx"] > 5]
        for tid in stale_gc:
            del self.graveyard_candidates[tid]

        # Decrement reseed grace
        grace = getattr(self, '_reseed_grace_frames', 0)
        if grace > 0:
            self._reseed_grace_frames = grace - 1

        self._update_group_distances()
        self._update_predicted_group_motion()

        return assignments

    def _handle_reseed(self, bboxes, hists, bios, embs, mots):
        """Handle identity-based reseed matching. Returns assignments list or None."""
        from scipy.optimize import linear_sum_assignment

        group_graves = [g_tid for g_tid in self.graveyard if g_tid in self._original_group_ids]
        active_group = [t for t in self.active_tracks if t in self._original_group_ids]
        logger.info(f"Reseed executing: {len(group_graves)} group in graveyard, "
                    f"{len(active_group)} group active, {len(bboxes)} dets")

        n_det = len(bboxes)
        n_grave = len(group_graves)
        reseed_assignments = {}
        assigned_dets = set()

        if group_graves:
            score_matrix = np.zeros((n_det, n_grave), dtype=np.float32)
            for di in range(n_det):
                for gi, g_tid in enumerate(group_graves):
                    identity = self._identity_score(hists[di], bios[di], embs[di],
                                                    g_tid, use_graveyard=True,
                                                    det_motion=mots[di])
                    g_avg = self.graveyard_area.get(g_tid, 0)
                    det_area = _bbox_area(bboxes[di])
                    size_ok = 1.0
                    if g_avg > 0 and det_area > 0:
                        ratio = det_area / g_avg if det_area >= g_avg else g_avg / det_area
                        if ratio > self.max_size_ratio:
                            size_ok = 0.0
                    score_matrix[di, gi] = identity * size_ok

            # Hungarian assignment (maximize score → minimize -score)
            if n_det > 0 and n_grave > 0:
                cost = -score_matrix
                row_idx, col_idx = linear_sum_assignment(cost)
                for r, c in zip(row_idx, col_idx):
                    if score_matrix[r, c] >= self.reseed_identity_threshold:
                        g_tid = group_graves[c]
                        reseed_assignments[r] = g_tid
                        assigned_dets.add(r)

                for di, g_tid in reseed_assignments.items():
                    self._recover_from_graveyard(g_tid, bboxes[di], hists[di], bios[di], embs[di])
                    logger.info(f"Reseed identity-matched detection {di} -> track {g_tid} "
                                f"(score={score_matrix[di, group_graves.index(g_tid)]:.3f})")

        # If group members disappeared from everywhere, restore to graveyard from seed
        missing_group = [t for t in self._original_group_ids
                         if t not in self.active_tracks and t not in self.graveyard]
        if missing_group:
            logger.warning(f"Group tracks {missing_group} vanished — restoring to graveyard from seed")
            for tid in missing_group:
                orig = self._original_bboxes.get(tid)
                if orig:
                    self.graveyard[tid] = orig
                    self.graveyard_area[tid] = _bbox_area(orig)
                    self.graveyard_age[tid] = self.relock_min_wait
                    self.graveyard_velocity[tid] = (0.0, 0.0)
                    if tid in self._original_histograms:
                        self.graveyard_histogram[tid] = list(self._original_histograms[tid])
            group_graves = [g_tid for g_tid in self.graveyard if g_tid in self._original_group_ids]

        if not reseed_assignments and group_graves:
            # Fallback: proximity to seed positions
            logger.warning("Identity-based reseed found no matches, falling back to proximity")
            for g_tid in group_graves:
                orig_bbox = self._original_bboxes.get(g_tid)
                if orig_bbox is None:
                    continue
                best_di = -1
                best_dist = 999.0
                for di in range(n_det):
                    if di in assigned_dets:
                        continue
                    d = centroid_distance(bboxes[di], orig_bbox)
                    if d < best_dist:
                        best_dist = d
                        best_di = di
                if best_di >= 0:
                    reseed_assignments[best_di] = g_tid
                    assigned_dets.add(best_di)
                    self._recover_from_graveyard(g_tid, bboxes[best_di], hists[best_di],
                                                 bios[best_di], embs[best_di])
                    logger.info(f"Reseed proximity-matched detection {best_di} -> track {g_tid} "
                                f"(dist={best_dist:.3f})")

            if reseed_assignments:
                self.group_ids = set(self._original_group_ids)
                self._update_group_distances()
                if len(reseed_assignments) == n_det:
                    self._frame_counter += 1
                    return [reseed_assignments.get(i, -1) for i in range(n_det)]

        return None  # Continue with normal update path

    def _try_graveyard_match(self, det_idx, bbox, hist, bio, emb, motion, assignments, all_bboxes):
        """Try to match a detection against graveyarded tracks. Returns dancer_id or None."""
        best_grave_tid = None
        best_score = -1
        det_area = _bbox_area(bbox)

        for g_tid, g_bbox in self.graveyard.items():
            if self.graveyard_age.get(g_tid, 0) < self.relock_min_wait:
                continue
            g_avg = self.graveyard_area.get(g_tid, 0)
            if g_avg <= 0 or det_area <= 0:
                continue
            ratio = det_area / g_avg if det_area >= g_avg else g_avg / det_area
            if ratio > self.max_size_ratio:
                continue

            dist = centroid_distance(bbox, g_bbox)
            dist_limit = self.max_centroid_dist * (4.0 if g_tid in self.group_ids else 2.0)
            if dist > dist_limit:
                continue

            prox = max(0, 1.0 - dist / (self.max_centroid_dist * 2.0))
            identity = self._identity_score(hist, bio, emb, g_tid,
                                            use_graveyard=True, det_motion=motion)
            score = prox * 0.20 + identity * 0.80
            if score > best_score:
                best_score = score
                best_grave_tid = g_tid

        fail_count = self.reid_fail_count.get(best_grave_tid, 0) if best_grave_tid is not None else 0
        reid_threshold = min(self.reid_base_threshold + 0.1 * fail_count, 0.8)

        if best_grave_tid is None or best_score <= reid_threshold:
            return None

        # Group coherence check
        if best_grave_tid in self.group_ids and len(self.group_distances) >= 1:
            id_score = self._identity_score(hist, bio, emb, best_grave_tid,
                                            use_graveyard=True, det_motion=motion)
            if id_score < 0.7:
                proposed = {assignments[d]: all_bboxes[d] for d in range(len(assignments))
                           if assignments[d] is not None and assignments[d] != -1}
                proposed[best_grave_tid] = bbox
                coherence = self._group_coherence_score(bbox, best_grave_tid, proposed)
                if coherence < self.coherence_threshold:
                    return None

        # 3-frame confirmation
        det_cx = (bbox[0] + bbox[2]) / 2
        det_cy = (bbox[1] + bbox[3]) / 2
        cand = self.graveyard_candidates.get(best_grave_tid)
        if cand and self._frame_counter - cand["frame_idx"] <= 2:
            cdist = ((det_cx - cand["cx"]) ** 2 + (det_cy - cand["cy"]) ** 2) ** 0.5
            if cdist < 0.1:
                cand["count"] += 1
                cand["cx"] = det_cx
                cand["cy"] = det_cy
                cand["frame_idx"] = self._frame_counter
            else:
                self.graveyard_candidates[best_grave_tid] = {
                    "cx": det_cx, "cy": det_cy, "count": 1,
                    "frame_idx": self._frame_counter,
                }
        else:
            self.graveyard_candidates[best_grave_tid] = {
                "cx": det_cx, "cy": det_cy, "count": 1,
                "frame_idx": self._frame_counter,
            }

        if self.graveyard_candidates.get(best_grave_tid, {}).get("count", 0) >= self.relock_confirm:
            self._recover_from_graveyard(best_grave_tid, bbox, hist, bio, emb)
            logger.info(f"Re-identified track {best_grave_tid} after occlusion "
                        f"(score={best_score:.2f}, threshold={reid_threshold:.2f})")
            return best_grave_tid

        return None

    def _try_identity_relock(self, det_idx, bbox, hist, bio, emb, motion, unrecovered_group):
        """Try identity-based re-lock for unmatched detections against group graves."""
        det_area = _bbox_area(bbox)
        best_g_tid = None
        best_id_score = -1

        for g_tid in unrecovered_group:
            g_avg = self.graveyard_area.get(g_tid, 0)
            if g_avg > 0 and det_area > 0:
                ratio = det_area / g_avg if det_area >= g_avg else g_avg / det_area
                if ratio > self.max_size_ratio:
                    continue
            elif g_avg <= 0 or det_area <= 0:
                continue
            id_score = self._identity_score(hist, bio, emb, g_tid,
                                            use_graveyard=True, det_motion=motion)
            if id_score > best_id_score:
                best_id_score = id_score
                best_g_tid = g_tid

        if best_g_tid is None or best_id_score <= 0.5:
            return None

        det_cx = (bbox[0] + bbox[2]) / 2
        det_cy = (bbox[1] + bbox[3]) / 2
        candidate = self.relock_candidates.get(best_g_tid)
        if candidate and self._frame_counter - candidate["frame_idx"] <= 2:
            cdist = ((det_cx - candidate["cx"]) ** 2 + (det_cy - candidate["cy"]) ** 2) ** 0.5
            if cdist < 0.1:
                candidate["count"] += 1
                candidate["cx"] = det_cx
                candidate["cy"] = det_cy
                candidate["frame_idx"] = self._frame_counter
            else:
                self.relock_candidates[best_g_tid] = {
                    "cx": det_cx, "cy": det_cy, "count": 1,
                    "frame_idx": self._frame_counter,
                }
        else:
            self.relock_candidates[best_g_tid] = {
                "cx": det_cx, "cy": det_cy, "count": 1,
                "frame_idx": self._frame_counter,
            }

        if self.relock_candidates.get(best_g_tid, {}).get("count", 0) >= 3:
            self._recover_from_graveyard(best_g_tid, bbox, hist, bio, emb)
            del self.relock_candidates[best_g_tid]
            logger.info(f"Re-locked track {best_g_tid} via identity (score={best_id_score:.2f})")
            return best_g_tid

        return None

    def _sync_from_bytetrack(self):
        """Sync active_tracks from ByteTrack state (used for empty-frame updates)."""
        # Increment missing counts for all active tracks
        new_active = {}
        new_missing = {}
        for dancer_id, bbox in self.active_tracks.items():
            age = self.missing_count.get(dancer_id, 0) + 1
            if age <= self.max_missing:
                # Get predicted position from ByteTrack Kalman filter
                bt_id = self._dancer_to_bt.get(dancer_id)
                pred_bbox = bbox
                if bt_id is not None:
                    for s in self._bytetrack.tracked_stracks + self._bytetrack.lost_stracks:
                        if s.track_id == bt_id:
                            pred_bbox = s.bbox
                            break
                new_active[dancer_id] = pred_bbox
                new_missing[dancer_id] = age
            else:
                seen = self.last_seen_bbox.get(dancer_id, bbox)
                self._move_to_graveyard(dancer_id, seen)

        self.active_tracks = new_active
        self.missing_count = new_missing

    def _age_graveyard(self):
        """Age out graveyard entries. Group members never expire."""
        expired = []
        for g_tid in self.graveyard:
            self.graveyard_age[g_tid] = self.graveyard_age.get(g_tid, 0) + 1
            if self.graveyard_age[g_tid] > self.graveyard_frames:
                if g_tid in self.group_ids:
                    continue
                expired.append(g_tid)
        for g_tid in expired:
            self._clear_graveyard_entry(g_tid)


def run_tracker(
    all_frames_detections: list[list[dict]],
    min_frame_ratio: float = 0.3,
) -> list[dict]:
    """Run tracker across detection frames and return stable person summaries.

    Args:
        all_frames_detections: List of frames, each frame is a list of person dicts
            with 'bbox' and 'dancer_pose' keys. Optionally 'color_histogram'.
        min_frame_ratio: Minimum fraction of frames a person must appear in to be kept.

    Returns:
        List of detected person dicts: {track_id, bbox, representative_pose, frame_count, area, appearance}
    """
    # Use BYTETracker directly for the detection pass (simpler, no identity needed)
    from app.pipeline.bytetrack import BYTETracker as _BT, STrack as _ST
    _ST.reset_id()

    bt = _BT(
        track_thresh=0.5,
        match_thresh=0.8,
        low_thresh=0.1,
        second_match_thresh=0.9,
        max_time_lost=30,  # shorter for detection pass
        min_hits=1,
    )

    # track_id -> list of (frame_idx, bbox, area, det_data)
    track_history: dict[int, list] = {}
    # Track histograms via EMA
    track_histograms: dict[int, list[float]] = {}

    for frame_idx, detections in enumerate(all_frames_detections):
        if not detections:
            bt.update([], [])
            continue

        bboxes = [d["bbox"] for d in detections]
        scores = [1.0] * len(bboxes)  # RTMPose doesn't provide per-detection confidence

        results = bt.update(bboxes, scores)

        for det_idx, tid in results:
            bbox = bboxes[det_idx]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append((frame_idx, bbox, area, detections[det_idx]))

            # Update histogram EMA
            hist = detections[det_idx].get("color_histogram")
            if hist:
                prev = track_histograms.get(tid)
                if not prev or len(prev) != len(hist):
                    track_histograms[tid] = list(hist)
                else:
                    alpha = 0.1
                    track_histograms[tid] = [
                        p * (1 - alpha) + h * alpha for p, h in zip(prev, hist)
                    ]

    # Filter tracks by minimum appearance
    min_frames = max(1, int(len(all_frames_detections) * min_frame_ratio))
    results_list = []

    for tid, history in track_history.items():
        if len(history) < min_frames:
            continue

        best_entry = max(history, key=lambda h: h[2])
        all_bboxes = np.array([h[1] for h in history])
        median_bbox = {
            "x_min": float(np.median(all_bboxes[:, 0])),
            "y_min": float(np.median(all_bboxes[:, 1])),
            "x_max": float(np.median(all_bboxes[:, 2])),
            "y_max": float(np.median(all_bboxes[:, 3])),
        }

        appearance = best_entry[3].get("appearance")

        result = {
            "track_id": tid,
            "bbox": median_bbox,
            "representative_pose": best_entry[3].get("dancer_pose", {}),
            "frame_count": len(history),
            "area": float(np.mean([h[2] for h in history])),
        }
        if appearance:
            result["appearance"] = appearance
        if tid in track_histograms:
            result["color_histogram"] = track_histograms[tid]

        results_list.append(result)

    results_list.sort(key=lambda r: -r["area"])
    logger.info(f"Tracker found {len(results_list)} stable persons from {len(track_history)} raw tracks")
    return results_list

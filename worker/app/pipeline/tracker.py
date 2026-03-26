"""IoU + centroid tracker with size gating, velocity prediction, appearance matching,
biometric body-proportion signatures, Re-ID embeddings, and graveyard re-identification
for robust occlusion recovery."""
import logging

import numpy as np

from app.pipeline.biometrics import (
    BiometricSignature, extract_biometric_signature, merge_signatures, signature_similarity,
)
from app.pipeline.reid import cosine_similarity as reid_cosine_similarity, merge_embeddings

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


def _shift_bbox(bbox: tuple, dx: float, dy: float) -> tuple:
    """Translate a bbox by (dx, dy)."""
    return (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)


class SimpleTracker:
    """Track persons across frames using IoU, centroid distance, appearance, and velocity.

    Occlusion handling:
    1. Size gating — rejects matches where bbox area ratio exceeds threshold
    2. Velocity prediction — predicts where each track should be, uses predicted
       bbox for matching during missing frames (handles walk-through occlusion where
       occluder travels at different speed/direction than dancer)
    3. Appearance matching — uses color histograms to prefer visually similar matches
    4. Extended memory — keeps tracks alive for max_missing frames using predicted position
    5. Graveyard re-ID — expired tracks recovered via position + size + appearance
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
        # Scale frame-based thresholds by FPS to maintain consistent time durations
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
        self.next_id = 0
        self.active_tracks: dict[int, tuple] = {}  # track_id -> last bbox (may be predicted)
        self.missing_count: dict[int, int] = {}  # track_id -> frames since last seen
        self.last_seen_bbox: dict[int, tuple] = {}  # track_id -> last actually-observed bbox
        self.track_avg_area: dict[int, float] = {}  # track_id -> running avg bbox area
        # Velocity: smoothed (dx, dy) per frame in normalized coords
        self.track_velocity: dict[int, tuple[float, float]] = {}
        # Appearance: color histogram per track
        self.track_histogram: dict[int, list[float]] = {}
        # Graveyard: tracks that expired from active but can be re-identified
        self.graveyard: dict[int, tuple] = {}  # track_id -> last bbox (predicted)
        self.graveyard_area: dict[int, float] = {}
        self.graveyard_age: dict[int, int] = {}
        self.graveyard_histogram: dict[int, list[float]] = {}
        self.graveyard_velocity: dict[int, tuple[float, float]] = {}
        # Group coherence: set of track IDs known to be part of the dancer group
        self.group_ids: set[int] = set()
        # Smoothed relative distances between group members (pair -> distance)
        self.group_distances: dict[tuple[int, int], float] = {}
        # Re-lock candidates: graveyard_tid -> {bbox, count, frame_idx}
        self.relock_candidates: dict[int, dict] = {}
        # Graveyard re-ID confirmation: require 3 frames at stable position
        self.graveyard_candidates: dict[int, dict] = {}  # g_tid -> {cx, cy, count, frame_idx, det_score}
        self.relock_confirm = 3
        # Re-ID failure tracking: count of short-lived re-IDs per track
        self.reid_fail_count: dict[int, int] = {}
        self.reid_frame: dict[int, int] = {}  # track_id -> frame when last re-ID'd
        # Biometric body-proportion signatures per track
        self.track_biometric: dict[int, BiometricSignature] = {}
        self.graveyard_biometric: dict[int, BiometricSignature] = {}
        # Re-ID CNN embeddings per track (512-dim L2-normalized)
        self.track_embedding: dict[int, np.ndarray] = {}
        self.graveyard_embedding: dict[int, np.ndarray] = {}
        # Reseed pending flag for identity-based reseed
        self._reseed_pending = False
        self._frame_counter = 0

    def seed(self, known_bboxes: dict[int, tuple], histograms: dict[int, list[float]] | None = None,
             group_ids: set[int] | None = None, biometrics: dict[int, BiometricSignature] | None = None,
             embeddings: dict[int, np.ndarray] | None = None):
        """Seed the tracker with known track_id -> bbox mappings from a prior detection pass."""
        self._original_bboxes = dict(known_bboxes)
        self._original_histograms = dict(histograms) if histograms else {}
        self._original_group_ids = set(group_ids) if group_ids else set(known_bboxes.keys())
        for tid, bbox in known_bboxes.items():
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
        # Initialize pairwise distances for the group
        self._update_group_distances()
        self.next_id = max(known_bboxes.keys()) + 1 if known_bboxes else 0

    def reseed(self):
        """Set reseed-pending flag for identity-based re-matching on the next update().

        Instead of reverting to frame-1 positions (which are wrong after dancers move),
        the next update() will use biometric + Re-ID + appearance signals to match
        incoming detections to the known dancer identities.

        Falls back to appearance-only matching if no identity signals are available.
        """
        if not hasattr(self, '_original_bboxes') or not self._original_bboxes:
            return
        # Move all active group members to graveyard so they can be identity-matched
        for tid in list(self.active_tracks.keys()):
            if tid in self._original_group_ids:
                seen = self.last_seen_bbox.get(tid, self.active_tracks[tid])
                self.graveyard[tid] = seen
                self.graveyard_area[tid] = self.track_avg_area.get(tid, 0)
                self.graveyard_age[tid] = self.relock_min_wait  # skip wait period
                self.graveyard_velocity[tid] = self.track_velocity.get(tid, (0.0, 0.0))
                if tid in self.track_histogram:
                    self.graveyard_histogram[tid] = self.track_histogram[tid]
                if tid in self.track_biometric:
                    self.graveyard_biometric[tid] = self.track_biometric[tid]
                if tid in self.track_embedding:
                    self.graveyard_embedding[tid] = self.track_embedding[tid]
                self.active_tracks.pop(tid, None)
                self.missing_count.pop(tid, None)
                self.graveyard_candidates.pop(tid, None)
                self.reid_fail_count.pop(tid, None)
                self.reid_frame.pop(tid, None)
        # Also ensure any existing graveyard entries for group members are ready
        for tid in self._original_group_ids:
            if tid in self.graveyard:
                self.graveyard_age[tid] = max(self.graveyard_age.get(tid, 0), self.relock_min_wait)
        self._reseed_pending = True
        self._reseed_grace_frames = int(10 * (self.effective_fps / 30.0))
        logger.info(f"Reseed pending — identity-based re-matching will run on next update "
                    f"for tracks {sorted(self._original_group_ids)}")

    def _update_group_distances(self):
        """Update smoothed pairwise centroid distances between group members."""
        gids = [tid for tid in self.group_ids if tid in self.active_tracks and self.missing_count.get(tid, 0) == 0]
        alpha = 0.15
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
        """Score how well a candidate assignment preserves the group's spatial formation.

        Returns 1.0 if fully coherent, lower values indicate the match breaks formation.
        A bystander walking through will have very different relative distances to other
        group members compared to what's expected.
        """
        if candidate_tid not in self.group_ids or len(self.group_distances) == 0:
            return 1.0  # No group info, don't penalize

        penalties = []
        for other_tid in self.group_ids:
            if other_tid == candidate_tid:
                continue
            pair = (min(candidate_tid, other_tid), max(candidate_tid, other_tid))
            expected_dist = self.group_distances.get(pair)
            if expected_dist is None or expected_dist <= 0:
                continue
            # Use proposed position for other if available, else current track position
            other_bbox = proposed_assignments.get(other_tid)
            if other_bbox is None:
                other_bbox = self.active_tracks.get(other_tid)
            if other_bbox is None:
                continue
            actual_dist = centroid_distance(candidate_bbox, other_bbox)
            # Ratio of actual to expected distance — 1.0 = perfect
            ratio = actual_dist / expected_dist if expected_dist > 0 else 1.0
            # Penalize deviations > 50% from expected
            if ratio > 1.5 or ratio < 0.5:
                penalties.append(abs(1.0 - ratio))

        if not penalties:
            return 1.0
        avg_penalty = sum(penalties) / len(penalties)
        return max(0.0, 1.0 - avg_penalty)

    def _predicted_bbox(self, track_id: int) -> tuple:
        """Return predicted bbox for a track based on last position + velocity × missed frames."""
        bbox = self.active_tracks.get(track_id)
        if bbox is None:
            return (0, 0, 0, 0)
        vel = self.track_velocity.get(track_id, (0.0, 0.0))
        missed = self.missing_count.get(track_id, 0)
        if missed == 0 or (vel[0] == 0 and vel[1] == 0):
            return bbox
        return _shift_bbox(bbox, vel[0] * missed, vel[1] * missed)

    def _effective_centroid_dist(self) -> float:
        grace = getattr(self, '_reseed_grace_frames', 0)
        if grace > 0:
            return self.max_centroid_dist * 3.0  # 0.9 normalized — nearly full frame
        return self.max_centroid_dist

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
        """Update running average area for a track (exponential moving average)."""
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
        reid_at = self.reid_frame.get(tid)
        if reid_at is not None:
            frames_since_reid = self._frame_counter - reid_at
            if frames_since_reid < self.max_missing * 2:
                self.reid_fail_count[tid] = self.reid_fail_count.get(tid, 0) + 1
                logger.info(f"Track {tid} re-ID was short-lived ({frames_since_reid} frames), "
                            f"fail_count now {self.reid_fail_count[tid]}")
            self.reid_frame.pop(tid, None)
        if tid in self.group_ids:
            logger.warning(f"GROUP TRACK {tid} moved to graveyard after {self.max_missing} missing frames "
                           f"(last seen at {_bbox_centroid(seen_bbox)})")
        else:
            logger.debug(f"Track {tid} moved to graveyard after {self.max_missing} missing frames")

    def _clear_graveyard_entry(self, g_tid: int):
        """Remove a graveyard entry and all associated identity data."""
        if g_tid in self.group_ids:
            logger.warning(f"GROUP TRACK {g_tid} EXPIRED from graveyard at frame {self._frame_counter} "
                           f"(after {self.graveyard_frames} frames)")
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

    def _identity_score(self, det_hist, det_bio, det_emb, track_id: int,
                        use_graveyard: bool = False) -> float:
        """Compute blended identity score using histogram + biometric + Re-ID signals.

        Weights adapt based on which signals are available.
        Target weights: proximity-free identity = 0.25 histogram + 0.25 biometric + 0.30 reid
        (proximity handled by caller).
        """
        scores = {}
        weights = {}

        # Histogram
        if use_graveyard:
            track_hist = self.graveyard_histogram.get(track_id)
        else:
            track_hist = self.track_histogram.get(track_id)
        if det_hist and track_hist and len(det_hist) == len(track_hist):
            scores["hist"] = float(sum(min(a, b) for a, b in zip(det_hist, track_hist)))
            weights["hist"] = 0.25
        else:
            scores["hist"] = 0.5
            weights["hist"] = 0.10

        # Biometric
        if use_graveyard:
            track_bio = self.graveyard_biometric.get(track_id)
        else:
            track_bio = self.track_biometric.get(track_id)
        bio_sim = signature_similarity(det_bio, track_bio)
        if bio_sim != 0.5 or (det_bio is not None and track_bio is not None):
            scores["bio"] = bio_sim
            weights["bio"] = 0.25
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
            weights["reid"] = 0.30
        else:
            scores["reid"] = 0.5
            weights["reid"] = 0.05

        total_w = sum(weights.values())
        if total_w < 1e-6:
            return 0.5
        return sum(scores[k] * weights[k] for k in scores) / total_w

    def update(self, bboxes: list[tuple], histograms: list[list[float]] | None = None,
               biometrics: list[BiometricSignature | None] | None = None,
               embeddings: list[np.ndarray | None] | None = None) -> list[int]:
        """Assign track IDs to a list of bounding boxes for one frame.

        Args:
            bboxes: List of (x_min, y_min, x_max, y_max) tuples.
            histograms: Optional parallel list of color histograms for appearance matching.
            biometrics: Optional parallel list of BiometricSignature for body-proportion matching.
            embeddings: Optional parallel list of Re-ID embeddings (512-dim) for visual identity matching.

        Returns a list of track_ids parallel to the input bboxes.
        """
        hists = histograms or [None] * len(bboxes)
        bios = biometrics or [None] * len(bboxes)
        embs = embeddings or [None] * len(bboxes)

        # Identity-based reseed: match detections to graveyarded group members
        if self._reseed_pending and bboxes:
            self._reseed_pending = False
            group_graves = [g_tid for g_tid in self.graveyard if g_tid in self._original_group_ids]
            if group_graves:
                # Build cost matrix: score each detection against each graveyarded group member
                n_det = len(bboxes)
                n_grave = len(group_graves)
                score_matrix = np.zeros((n_det, n_grave), dtype=np.float32)
                for di in range(n_det):
                    for gi, g_tid in enumerate(group_graves):
                        identity = self._identity_score(hists[di], bios[di], embs[di],
                                                        g_tid, use_graveyard=True)
                        # Also factor in size compatibility
                        g_avg = self.graveyard_area.get(g_tid, 0)
                        det_area = _bbox_area(bboxes[di])
                        size_ok = 1.0
                        if g_avg > 0 and det_area > 0:
                            ratio = det_area / g_avg if det_area >= g_avg else g_avg / det_area
                            if ratio > self.max_size_ratio:
                                size_ok = 0.0
                        score_matrix[di, gi] = identity * size_ok

                # Greedy assignment: best score first
                assigned_dets = set()
                assigned_graves = set()
                flat_scores = []
                for di in range(n_det):
                    for gi in range(n_grave):
                        flat_scores.append((score_matrix[di, gi], di, gi))
                flat_scores.sort(reverse=True)

                reseed_assignments = {}
                for score, di, gi in flat_scores:
                    if di in assigned_dets or gi in assigned_graves:
                        continue
                    if score < 0.4:
                        break
                    g_tid = group_graves[gi]
                    reseed_assignments[di] = g_tid
                    assigned_dets.add(di)
                    assigned_graves.add(gi)

                # Recover matched tracks from graveyard
                for di, g_tid in reseed_assignments.items():
                    self.active_tracks[g_tid] = bboxes[di]
                    self.missing_count[g_tid] = 0
                    self.last_seen_bbox[g_tid] = bboxes[di]
                    self._update_avg_area(g_tid, bboxes[di])
                    self.track_velocity[g_tid] = (0.0, 0.0)
                    if g_tid in self.graveyard_histogram:
                        self.track_histogram[g_tid] = self.graveyard_histogram.pop(g_tid)
                    if g_tid in self.graveyard_biometric:
                        self.track_biometric[g_tid] = self.graveyard_biometric.pop(g_tid)
                    if g_tid in self.graveyard_embedding:
                        self.track_embedding[g_tid] = self.graveyard_embedding.pop(g_tid)
                    self.graveyard.pop(g_tid, None)
                    self.graveyard_area.pop(g_tid, None)
                    self.graveyard_age.pop(g_tid, None)
                    self.graveyard_velocity.pop(g_tid, None)
                    self._update_histogram(g_tid, hists[di])
                    self._update_biometric(g_tid, bios[di])
                    self._update_embedding(g_tid, embs[di])
                    logger.info(f"Reseed identity-matched detection {di} -> track {g_tid} "
                                f"(score={score_matrix[di, list(group_graves).index(g_tid)]:.3f})")

                if reseed_assignments:
                    self.group_ids = set(self._original_group_ids)
                    self._update_group_distances()
                    # Return early with assignments if all detections matched
                    if len(reseed_assignments) == n_det:
                        self._frame_counter += 1
                        return [reseed_assignments.get(i, -1) for i in range(n_det)]
                    # Otherwise fall through with unmatched detections handled normally

                if not reseed_assignments:
                    # Fallback: no identity signals available, use widened distance
                    logger.warning("Identity-based reseed found no matches, falling back to distance-based matching")

        if not self.active_tracks:
            # First frame (or all tracks expired): assign new IDs
            ids = []
            for i, bbox in enumerate(bboxes):
                ids.append(self.next_id)
                self.active_tracks[self.next_id] = bbox
                self.missing_count[self.next_id] = 0
                self.track_avg_area[self.next_id] = _bbox_area(bbox)
                self.track_velocity[self.next_id] = (0.0, 0.0)
                if hists[i]:
                    self.track_histogram[self.next_id] = hists[i]
                self.next_id += 1
            return ids

        # No detections this frame: all active tracks become missing
        if not bboxes:
            new_tracks = {}
            new_missing = {}
            for tid in list(self.active_tracks.keys()):
                age = self.missing_count.get(tid, 0) + 1
                if age <= self.max_missing:
                    vel = self.track_velocity.get(tid, (0.0, 0.0))
                    predicted = _shift_bbox(self.active_tracks[tid], vel[0], vel[1])
                    new_tracks[tid] = predicted
                    new_missing[tid] = age
                else:
                    seen = self.last_seen_bbox.get(tid, self.active_tracks[tid])
                    self._move_to_graveyard(tid, seen)
            # Age out graveyard entries
            expired_graves = []
            for g_tid in self.graveyard:
                self.graveyard_age[g_tid] = self.graveyard_age.get(g_tid, 0) + 1
                if self.graveyard_age[g_tid] > self.graveyard_frames:
                    expired_graves.append(g_tid)
            for g_tid in expired_graves:
                self._clear_graveyard_entry(g_tid)
            self.active_tracks = new_tracks
            self.missing_count = new_missing
            self._frame_counter += 1
            return []

        # Build predicted bboxes for tracks that have been missing
        track_ids = list(self.active_tracks.keys())
        track_bboxes = []
        for tid in track_ids:
            if self.missing_count.get(tid, 0) > 0:
                track_bboxes.append(self._predicted_bbox(tid))
            else:
                track_bboxes.append(self.active_tracks[tid])

        used_tracks = set()
        assignments = [None] * len(bboxes)

        # Sort detections by area (largest first for priority)
        det_order = sorted(range(len(bboxes)), key=lambda i: -_bbox_area(bboxes[i]))

        for det_idx in det_order:
            best_score = -1
            best_tid_idx = -1

            for tid_idx, tid in enumerate(track_ids):
                if tid in used_tracks:
                    continue
                if not self._size_compatible(bboxes[det_idx], tid):
                    continue
                iou = compute_iou(bboxes[det_idx], track_bboxes[tid_idx])
                if iou < self.iou_threshold:
                    continue
                # Blend IoU with appearance similarity
                app_sim = self._appearance_similarity(hists[det_idx], tid)
                w = self.appearance_weight
                score = iou * (1 - w) + app_sim * w
                if score > best_score:
                    best_score = score
                    best_tid_idx = tid_idx

            # Fallback to centroid distance (against predicted position)
            if best_tid_idx == -1:
                best_dist = self._effective_centroid_dist()
                best_app = -1
                for tid_idx, tid in enumerate(track_ids):
                    if tid in used_tracks:
                        continue
                    if not self._size_compatible(bboxes[det_idx], tid):
                        continue
                    dist = centroid_distance(bboxes[det_idx], track_bboxes[tid_idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_tid_idx = tid_idx
                    elif dist == best_dist and best_tid_idx >= 0:
                        # Tie-break by appearance
                        app = self._appearance_similarity(hists[det_idx], tid)
                        if app > best_app:
                            best_app = app
                            best_tid_idx = tid_idx

            if best_tid_idx >= 0:
                tid = track_ids[best_tid_idx]
                # Group coherence check: if this track is a known dancer,
                # verify the match doesn't break the group's spatial formation.
                # This catches cases where an occluder overlaps a dancer's predicted
                # position but has wrong relative distances to other group members.
                if tid in self.group_ids and len(self.group_distances) >= 1:
                    proposed = {t: bboxes[d] for d, t in enumerate(assignments) if t is not None}
                    proposed[tid] = bboxes[det_idx]
                    coherence = self._group_coherence_score(bboxes[det_idx], tid, proposed)
                    if coherence < 0.4:
                        # This detection breaks the formation — likely an occluder
                        logger.debug(f"Rejected match det {det_idx}->track {tid}: coherence={coherence:.2f}")
                        best_tid_idx = -1

            if best_tid_idx >= 0:
                assignments[det_idx] = track_ids[best_tid_idx]
                used_tracks.add(track_ids[best_tid_idx])

        # Positional fallback with size check
        unmatched_dets = [i for i in range(len(bboxes)) if assignments[i] is None]
        unmatched_tids = [tid for tid in track_ids if tid not in used_tracks]

        if unmatched_dets and unmatched_tids and len(unmatched_dets) == len(unmatched_tids):
            compat_dets = [i for i in unmatched_dets
                          if any(self._size_compatible(bboxes[i], tid) for tid in unmatched_tids)]
            compat_tids = [tid for tid in unmatched_tids
                          if any(self._size_compatible(bboxes[i], tid) for i in unmatched_dets)]
            if compat_dets and compat_tids and len(compat_dets) == len(compat_tids):
                compat_dets.sort(key=lambda i: (bboxes[i][0] + bboxes[i][2]) / 2)
                compat_tids.sort(key=lambda tid: (self.active_tracks[tid][0] + self.active_tracks[tid][2]) / 2)
                for det_idx, tid in zip(compat_dets, compat_tids):
                    if self._size_compatible(bboxes[det_idx], tid):
                        assignments[det_idx] = tid
                        used_tracks.add(tid)
            unmatched_dets = [i for i in range(len(bboxes)) if assignments[i] is None]
            unmatched_tids = [tid for tid in track_ids if tid not in used_tracks]

        # Re-identification: check unmatched detections against graveyard
        # (only consider graveyard entries that have waited long enough)
        still_unmatched = []
        for det_idx in unmatched_dets:
            best_grave_tid = None
            best_score = -1
            det_area = _bbox_area(bboxes[det_idx])
            for g_tid, g_bbox in self.graveyard.items():
                if self.graveyard_age.get(g_tid, 0) < self.relock_min_wait:
                    continue
                g_avg = self.graveyard_area.get(g_tid, 0)
                if g_avg <= 0 or det_area <= 0:
                    continue
                ratio = det_area / g_avg if det_area >= g_avg else g_avg / det_area
                if ratio > self.max_size_ratio:
                    continue
                # Use last known position — velocity extrapolation over hundreds
                # of frames causes predicted positions to fly off-screen.
                # For group members, also use a wider distance threshold since
                # dancers may have moved significantly during occlusion.
                dist = centroid_distance(bboxes[det_idx], g_bbox)
                dist_limit = self.max_centroid_dist * (4.0 if g_tid in self.group_ids else 2.0)
                if dist > dist_limit:
                    continue
                # Score: proximity + identity (histogram + biometric + Re-ID)
                prox = max(0, 1.0 - dist / (self.max_centroid_dist * 2.0))
                identity = self._identity_score(hists[det_idx], bios[det_idx], embs[det_idx],
                                                g_tid, use_graveyard=True)
                score = prox * 0.20 + identity * 0.80
                if score > best_score:
                    best_score = score
                    best_grave_tid = g_tid

            # Progressive threshold: penalize tracks with prior failed re-IDs
            fail_count = self.reid_fail_count.get(best_grave_tid, 0) if best_grave_tid is not None else 0
            reid_threshold = min(0.5 + 0.1 * fail_count, 0.8)

            if best_grave_tid is not None and best_score > reid_threshold:
                # Group coherence check: verify re-ID preserves spatial formation
                if best_grave_tid in self.group_ids and len(self.group_distances) >= 1:
                    proposed = {t: bboxes[d] for d, t in enumerate(assignments) if t is not None}
                    proposed[best_grave_tid] = bboxes[det_idx]
                    coherence = self._group_coherence_score(bboxes[det_idx], best_grave_tid, proposed)
                    if coherence < 0.4:
                        logger.debug(f"Rejected graveyard re-ID det {det_idx}->track {best_grave_tid}: "
                                     f"coherence={coherence:.2f}, score={best_score:.2f}")
                        still_unmatched.append(det_idx)
                        continue

                # 3-frame position-stable confirmation before accepting re-ID
                det_cx = (bboxes[det_idx][0] + bboxes[det_idx][2]) / 2
                det_cy = (bboxes[det_idx][1] + bboxes[det_idx][3]) / 2
                cand = self.graveyard_candidates.get(best_grave_tid)
                if cand and self._frame_counter - cand["frame_idx"] <= 2:
                    cdist = ((det_cx - cand["cx"]) ** 2 + (det_cy - cand["cy"]) ** 2) ** 0.5
                    if cdist < 0.1:
                        cand["count"] += 1
                        cand["cx"] = det_cx
                        cand["cy"] = det_cy
                        cand["frame_idx"] = self._frame_counter
                        cand["det_score"] = best_score
                    else:
                        self.graveyard_candidates[best_grave_tid] = {
                            "cx": det_cx, "cy": det_cy, "count": 1,
                            "frame_idx": self._frame_counter, "det_score": best_score,
                        }
                else:
                    self.graveyard_candidates[best_grave_tid] = {
                        "cx": det_cx, "cy": det_cy, "count": 1,
                        "frame_idx": self._frame_counter, "det_score": best_score,
                    }

                if self.graveyard_candidates.get(best_grave_tid, {}).get("count", 0) >= self.relock_confirm:
                    assignments[det_idx] = best_grave_tid
                    self.track_avg_area[best_grave_tid] = self.graveyard_area.pop(best_grave_tid)
                    self.track_velocity[best_grave_tid] = self.graveyard_velocity.pop(best_grave_tid, (0.0, 0.0))
                    if best_grave_tid in self.graveyard_histogram:
                        self.track_histogram[best_grave_tid] = self.graveyard_histogram.pop(best_grave_tid)
                    if best_grave_tid in self.graveyard_biometric:
                        self.track_biometric[best_grave_tid] = self.graveyard_biometric.pop(best_grave_tid)
                    if best_grave_tid in self.graveyard_embedding:
                        self.track_embedding[best_grave_tid] = self.graveyard_embedding.pop(best_grave_tid)
                    del self.graveyard[best_grave_tid]
                    del self.graveyard_age[best_grave_tid]
                    self.graveyard_candidates.pop(best_grave_tid, None)
                    self.reid_frame[best_grave_tid] = self._frame_counter
                    logger.info(f"Re-identified track {best_grave_tid} after occlusion "
                                f"(score={best_score:.2f}, threshold={reid_threshold:.2f}, fails={fail_count})")
                else:
                    still_unmatched.append(det_idx)
            else:
                still_unmatched.append(det_idx)

        # Identity re-lock: for group members in the graveyard that weren't
        # recovered by position, try matching on identity (appearance + biometric + Re-ID)
        # with 3-frame confirmation.
        relock_recovered = []
        unrecovered_group_graves = [
            g_tid for g_tid in self.graveyard
            if g_tid in self.group_ids
            and self.graveyard_age.get(g_tid, 0) >= self.relock_min_wait
            and g_tid not in {assignments[i] for i in range(len(assignments)) if assignments[i] is not None}
        ]
        if still_unmatched and unrecovered_group_graves:
            for det_idx in still_unmatched:
                det_area = _bbox_area(bboxes[det_idx])
                best_g_tid = None
                best_id_score = -1
                for g_tid in unrecovered_group_graves:
                    # Size check against graveyard area
                    g_avg = self.graveyard_area.get(g_tid, 0)
                    if g_avg > 0 and det_area > 0:
                        ratio = det_area / g_avg if det_area >= g_avg else g_avg / det_area
                        if ratio > self.max_size_ratio:
                            continue
                    elif g_avg <= 0 or det_area <= 0:
                        continue
                    id_score = self._identity_score(hists[det_idx], bios[det_idx], embs[det_idx],
                                                    g_tid, use_graveyard=True)
                    if id_score > best_id_score:
                        best_id_score = id_score
                        best_g_tid = g_tid

                if best_g_tid is not None and best_id_score > 0.5:
                    candidate = self.relock_candidates.get(best_g_tid)
                    det_cx = (bboxes[det_idx][0] + bboxes[det_idx][2]) / 2
                    det_cy = (bboxes[det_idx][1] + bboxes[det_idx][3]) / 2
                    if candidate and self._frame_counter - candidate["frame_idx"] <= 2:
                        cdist = ((det_cx - candidate["cx"]) ** 2 + (det_cy - candidate["cy"]) ** 2) ** 0.5
                        if cdist < 0.1:
                            candidate["count"] += 1
                            candidate["cx"] = det_cx
                            candidate["cy"] = det_cy
                            candidate["frame_idx"] = self._frame_counter
                            candidate["det_idx"] = det_idx
                        else:
                            self.relock_candidates[best_g_tid] = {
                                "cx": det_cx, "cy": det_cy, "count": 1,
                                "frame_idx": self._frame_counter, "det_idx": det_idx,
                            }
                    else:
                        self.relock_candidates[best_g_tid] = {
                            "cx": det_cx, "cy": det_cy, "count": 1,
                            "frame_idx": self._frame_counter, "det_idx": det_idx,
                        }

                    if self.relock_candidates.get(best_g_tid, {}).get("count", 0) >= 3:
                        assignments[det_idx] = best_g_tid
                        self.track_avg_area[best_g_tid] = self.graveyard_area.pop(best_g_tid)
                        self.track_velocity[best_g_tid] = (0.0, 0.0)
                        if best_g_tid in self.graveyard_histogram:
                            self.track_histogram[best_g_tid] = self.graveyard_histogram.pop(best_g_tid)
                        if best_g_tid in self.graveyard_biometric:
                            self.track_biometric[best_g_tid] = self.graveyard_biometric.pop(best_g_tid)
                        if best_g_tid in self.graveyard_embedding:
                            self.track_embedding[best_g_tid] = self.graveyard_embedding.pop(best_g_tid)
                        del self.graveyard[best_g_tid]
                        del self.graveyard_age[best_g_tid]
                        self.graveyard_velocity.pop(best_g_tid, None)
                        del self.relock_candidates[best_g_tid]
                        unrecovered_group_graves.remove(best_g_tid)
                        relock_recovered.append(det_idx)
                        self.reid_frame[best_g_tid] = self._frame_counter
                        logger.info(f"Re-locked track {best_g_tid} via identity "
                                    f"(score={best_id_score:.2f})")

        still_unmatched = [i for i in still_unmatched if i not in relock_recovered]

        # Assign new IDs to remaining unmatched detections
        for det_idx in still_unmatched:
            assignments[det_idx] = self.next_id
            self.next_id += 1

        # Update state
        new_tracks = {}
        new_missing = {}
        for det_idx, tid in enumerate(assignments):
            old_bbox = self.active_tracks.get(tid)
            new_tracks[tid] = bboxes[det_idx]
            new_missing[tid] = 0
            self.last_seen_bbox[tid] = bboxes[det_idx]
            self._update_avg_area(tid, bboxes[det_idx])
            self._update_histogram(tid, hists[det_idx])
            self._update_biometric(tid, bios[det_idx])
            self._update_embedding(tid, embs[det_idx])
            if old_bbox is not None:
                self._update_velocity(tid, old_bbox, bboxes[det_idx])
            elif tid not in self.track_velocity:
                self.track_velocity[tid] = (0.0, 0.0)

        # Keep unmatched tracks alive using predicted position
        for tid in unmatched_tids:
            age = self.missing_count.get(tid, 0) + 1
            if age <= self.max_missing:
                # Advance predicted position
                vel = self.track_velocity.get(tid, (0.0, 0.0))
                predicted = _shift_bbox(self.active_tracks[tid], vel[0], vel[1])
                new_tracks[tid] = predicted
                new_missing[tid] = age
            else:
                # Move to graveyard — use last *observed* position, not the
                # velocity-drifted one which may be far off-screen.
                seen = self.last_seen_bbox.get(tid, self.active_tracks[tid])
                self._move_to_graveyard(tid, seen)

        # Age out graveyard entries
        expired_graves = []
        for g_tid in self.graveyard:
            self.graveyard_age[g_tid] = self.graveyard_age.get(g_tid, 0) + 1
            if self.graveyard_age[g_tid] > self.graveyard_frames:
                expired_graves.append(g_tid)
        for g_tid in expired_graves:
            self._clear_graveyard_entry(g_tid)

        self.active_tracks = new_tracks
        self.missing_count = new_missing
        self._frame_counter += 1

        # Clean stale relock and graveyard candidates (not seen for >5 frames)
        stale = [tid for tid, c in self.relock_candidates.items()
                 if self._frame_counter - c["frame_idx"] > 5]
        for tid in stale:
            del self.relock_candidates[tid]
        stale_gc = [tid for tid, c in self.graveyard_candidates.items()
                    if self._frame_counter - c["frame_idx"] > 5]
        for tid in stale_gc:
            del self.graveyard_candidates[tid]

        # Decrement reseed grace period
        grace = getattr(self, '_reseed_grace_frames', 0)
        if grace > 0:
            self._reseed_grace_frames = grace - 1

        # Update group formation distances (only from visible group members)
        self._update_group_distances()

        return assignments


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
    tracker = SimpleTracker()
    # track_id -> list of (frame_idx, bbox, area, pose_data)
    track_history: dict[int, list] = {}

    for frame_idx, detections in enumerate(all_frames_detections):
        bboxes = [d["bbox"] for d in detections]
        if not bboxes:
            continue

        # Pass histograms if available
        hists = [d.get("color_histogram") for d in detections]
        has_hists = any(h is not None for h in hists)
        track_ids = tracker.update(bboxes, histograms=hists if has_hists else None)

        for det_idx, tid in enumerate(track_ids):
            bbox = bboxes[det_idx]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append((frame_idx, bbox, area, detections[det_idx]))

    # Filter tracks by minimum appearance
    min_frames = max(1, int(len(all_frames_detections) * min_frame_ratio))
    results = []

    for tid, history in track_history.items():
        if len(history) < min_frames:
            continue

        # Pick the frame where this person had the largest bounding box area
        best_entry = max(history, key=lambda h: h[2])
        # Compute median bbox
        all_bboxes = np.array([h[1] for h in history])
        median_bbox = {
            "x_min": float(np.median(all_bboxes[:, 0])),
            "y_min": float(np.median(all_bboxes[:, 1])),
            "x_max": float(np.median(all_bboxes[:, 2])),
            "y_max": float(np.median(all_bboxes[:, 3])),
        }

        # Aggregate appearance from the track's histogram
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
        # Also pass the tracker's averaged histogram
        if tid in tracker.track_histogram:
            result["color_histogram"] = tracker.track_histogram[tid]

        results.append(result)

    # Sort by area descending (largest person first)
    results.sort(key=lambda r: -r["area"])
    logger.info(f"Tracker found {len(results)} stable persons from {len(track_history)} raw tracks")
    return results

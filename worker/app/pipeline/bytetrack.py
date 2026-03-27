"""ByteTrack multi-object tracker with Kalman filter prediction.

Implements the core ByteTrack algorithm (Zhang et al., ECCV 2022):
- Two-pass association: high-confidence then low-confidence detections
- Kalman filter with constant-velocity model for motion prediction
- Hungarian (linear_sum_assignment) for optimal matching
- Works with normalized coordinates (0-1 range)

This module handles the MOT problem: which detection matches which detection
across frames. Identity (which person is which dancer) is handled by the
caller (SimpleTracker wrapper in tracker.py).
"""
import logging
from enum import IntEnum
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class TrackState(IntEnum):
    Tracked = 0
    Lost = 1
    Removed = 2


def _iou_batch(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes.

    Args:
        atlbrs: (N, 4) array of [x1, y1, x2, y2]
        btlbrs: (M, 4) array of [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    a = np.asarray(atlbrs, dtype=np.float64)
    b = np.asarray(btlbrs, dtype=np.float64)

    # Intersection
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)  # (N, M)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Union
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # (M,)
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0).astype(np.float32)


class KalmanFilter:
    """Simple constant-velocity Kalman filter for bounding box tracking.

    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]

    All coordinates are in normalized (0-1) space.
    """

    # Shared matrices (class-level, allocated once)
    _F = np.eye(8, dtype=np.float64)
    _F[0, 4] = 1.0  # cx += vx * dt
    _F[1, 5] = 1.0  # cy += vy * dt
    _F[2, 6] = 1.0  # w  += vw * dt
    _F[3, 7] = 1.0  # h  += vh * dt

    _H = np.zeros((4, 8), dtype=np.float64)
    _H[0, 0] = 1.0
    _H[1, 1] = 1.0
    _H[2, 2] = 1.0
    _H[3, 3] = 1.0

    def __init__(self):
        # Process noise — position components get less noise, velocity gets more
        self._Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4,
                           2.5e-3, 2.5e-3, 1e-3, 1e-3]).astype(np.float64)
        # Measurement noise
        self._R = np.diag([1e-3, 1e-3, 1e-3, 1e-3]).astype(np.float64)

        self.x = np.zeros(8, dtype=np.float64)  # state
        self.P = np.eye(8, dtype=np.float64) * 0.1  # covariance

    def init(self, measurement: np.ndarray):
        """Initialize state from first measurement [cx, cy, w, h]."""
        self.x[:4] = measurement
        self.x[4:] = 0.0  # zero initial velocity
        self.P = np.eye(8, dtype=np.float64) * 0.1
        # Higher initial uncertainty for velocity
        self.P[4, 4] = self.P[5, 5] = self.P[6, 6] = self.P[7, 7] = 1.0

    def predict(self):
        """Predict next state. Returns predicted [cx, cy, w, h]."""
        F = self._F
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q
        # Enforce minimum size
        self.x[2] = max(self.x[2], 1e-4)
        self.x[3] = max(self.x[3], 1e-4)
        return self.x[:4].copy()

    def update(self, measurement: np.ndarray):
        """Update state with measurement [cx, cy, w, h]."""
        H = self._H
        y = measurement - H @ self.x  # innovation
        S = H @ self.P @ H.T + self._R  # innovation covariance
        try:
            K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        except np.linalg.LinAlgError:
            # Singular matrix — skip update
            return
        self.x = self.x + K @ y
        I_KH = np.eye(8) - K @ H
        self.P = I_KH @ self.P
        # Enforce minimum size
        self.x[2] = max(self.x[2], 1e-4)
        self.x[3] = max(self.x[3], 1e-4)


def _bbox_to_cxywh(bbox: tuple | np.ndarray) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float64)


def _cxywh_to_bbox(cxywh: np.ndarray) -> tuple:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = cxywh[0], cxywh[1], cxywh[2], cxywh[3]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class STrack:
    """Single object track with Kalman filter state."""

    _next_id = 0

    @staticmethod
    def reset_id():
        STrack._next_id = 0

    @staticmethod
    def set_next_id(n: int):
        STrack._next_id = n

    def __init__(self, bbox: tuple, score: float = 1.0):
        self.kf = KalmanFilter()
        self.kf.init(_bbox_to_cxywh(bbox))

        self.track_id = STrack._next_id
        STrack._next_id += 1

        self.score = score
        self.state = TrackState.Tracked
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0
        self._bbox = bbox  # cache last bbox

    @property
    def bbox(self) -> tuple:
        """Current bbox as (x1, y1, x2, y2)."""
        return self._bbox

    def predict(self):
        """Run Kalman predict step."""
        pred = self.kf.predict()
        self._bbox = _cxywh_to_bbox(pred)

    def update(self, bbox: tuple, score: float, frame_id: int):
        """Update track with matched detection."""
        self.kf.update(_bbox_to_cxywh(bbox))
        self._bbox = bbox
        self.score = score
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.time_since_update = 0

    def activate(self, frame_id: int):
        """Activate a new track."""
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.state = TrackState.Tracked
        self.time_since_update = 0

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


def _linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Run Hungarian algorithm on cost matrix.

    Args:
        cost_matrix: (N, M) cost matrix (lower is better)
        thresh: maximum cost to accept a match

    Returns:
        matches: list of (row, col) tuples
        unmatched_rows: list of unmatched row indices
        unmatched_cols: list of unmatched col indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    # scipy.optimize.linear_sum_assignment minimizes total cost
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] > thresh:
            continue
        matches.append((r, c))
        unmatched_rows.discard(r)
        unmatched_cols.discard(c)

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)


class BYTETracker:
    """ByteTrack multi-object tracker.

    Two-pass association:
    1. Match high-confidence detections to tracked/lost tracks using IoU
    2. Match low-confidence detections to remaining unmatched tracks using IoU

    Args:
        track_thresh: confidence threshold to split high/low detections
        match_thresh: IoU cost threshold for first association (high-conf)
        low_thresh: minimum confidence to consider a detection at all
        second_match_thresh: IoU cost threshold for second association (low-conf)
        max_time_lost: frames a track survives without matches before removal
        min_hits: minimum consecutive hits before a track is considered confirmed
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        low_thresh: float = 0.1,
        second_match_thresh: float = 0.5,
        max_time_lost: int = 90,
        min_hits: int = 1,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.second_match_thresh = second_match_thresh
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits

        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0

    @property
    def all_stracks(self) -> list[STrack]:
        """All active tracks (tracked + lost)."""
        return self.tracked_stracks + self.lost_stracks

    def _get_strack_by_id(self, track_id: int) -> Optional[STrack]:
        """Find a track by ID across all pools."""
        for s in self.tracked_stracks + self.lost_stracks + self.removed_stracks:
            if s.track_id == track_id:
                return s
        return None

    def add_track(self, bbox: tuple, track_id: int, score: float = 1.0):
        """Manually add a track with a specific ID (used for seeding)."""
        strack = STrack.__new__(STrack)
        strack.kf = KalmanFilter()
        strack.kf.init(_bbox_to_cxywh(bbox))
        strack.track_id = track_id
        strack.score = score
        strack.state = TrackState.Tracked
        strack.is_activated = True
        strack.frame_id = self.frame_id
        strack.start_frame = self.frame_id
        strack.time_since_update = 0
        strack._bbox = bbox
        self.tracked_stracks.append(strack)

    def update(self, bboxes: list[tuple], scores: list[float] | None = None) -> list[tuple[int, int]]:
        """Process one frame of detections.

        Args:
            bboxes: list of (x1, y1, x2, y2) bounding boxes
            scores: confidence scores per detection (default 1.0 for all)

        Returns:
            List of (detection_index, track_id) for matched detections.
            Unmatched detections that create new tracks are also included.
        """
        self.frame_id += 1

        if scores is None:
            scores = [1.0] * len(bboxes)

        # ---- Step 1: predict all existing tracks ----
        for strack in self.tracked_stracks + self.lost_stracks:
            strack.predict()
            strack.time_since_update += 1

        # ---- Step 2: split detections into high-conf and low-conf ----
        det_bboxes = np.array(bboxes) if bboxes else np.zeros((0, 4))
        det_scores = np.array(scores)

        high_mask = det_scores >= self.track_thresh
        low_mask = (det_scores >= self.low_thresh) & (~high_mask)

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        high_bboxes = det_bboxes[high_indices] if len(high_indices) > 0 else np.zeros((0, 4))
        low_bboxes = det_bboxes[low_indices] if len(low_indices) > 0 else np.zeros((0, 4))

        # ---- Step 3: First association — high-conf detections vs tracked tracks ----
        # Combine tracked + recently lost tracks for matching
        strack_pool = self.tracked_stracks + self.lost_stracks

        if len(high_bboxes) > 0 and len(strack_pool) > 0:
            track_bboxes = np.array([s.bbox for s in strack_pool])
            iou_matrix = _iou_batch(high_bboxes, track_bboxes)
            cost_matrix = 1.0 - iou_matrix  # cost = 1 - IoU

            matches_first, unmatched_dets_first, unmatched_tracks_first = \
                _linear_assignment(cost_matrix, self.match_thresh)
        else:
            matches_first = []
            unmatched_dets_first = list(range(len(high_indices)))
            unmatched_tracks_first = list(range(len(strack_pool)))

        # Apply first-pass matches
        results = []
        matched_track_indices = set()
        for d_idx, t_idx in matches_first:
            det_orig_idx = int(high_indices[d_idx])
            strack = strack_pool[t_idx]
            strack.update(bboxes[det_orig_idx], scores[det_orig_idx], self.frame_id)
            results.append((det_orig_idx, strack.track_id))
            matched_track_indices.add(t_idx)

        # ---- Step 4: Second association — low-conf detections vs remaining tracked tracks ----
        # Only match against tracks that are currently Tracked (not Lost) and unmatched
        remaining_tracked = [
            (i, strack_pool[i])
            for i in unmatched_tracks_first
            if i < len(strack_pool) and strack_pool[i].state == TrackState.Tracked
        ]

        if len(low_bboxes) > 0 and len(remaining_tracked) > 0:
            r_track_bboxes = np.array([s.bbox for _, s in remaining_tracked])
            iou_matrix2 = _iou_batch(low_bboxes, r_track_bboxes)
            cost_matrix2 = 1.0 - iou_matrix2

            matches_second, unmatched_dets_second, unmatched_tracks_second = \
                _linear_assignment(cost_matrix2, self.second_match_thresh)

            for d_idx, t_idx in matches_second:
                det_orig_idx = int(low_indices[d_idx])
                pool_idx, strack = remaining_tracked[t_idx]
                strack.update(bboxes[det_orig_idx], scores[det_orig_idx], self.frame_id)
                results.append((det_orig_idx, strack.track_id))
                matched_track_indices.add(pool_idx)

            # Mark unmatched tracked stracks from second pass as lost
            for t_idx in unmatched_tracks_second:
                pool_idx, strack = remaining_tracked[t_idx]
                if strack.state == TrackState.Tracked:
                    strack.mark_lost()
        else:
            # No second pass possible — mark all remaining tracked as lost
            for i in unmatched_tracks_first:
                if i < len(strack_pool) and strack_pool[i].state == TrackState.Tracked:
                    if i not in matched_track_indices:
                        strack_pool[i].mark_lost()

        # Mark remaining unmatched Lost tracks — increment their missing time
        for i in unmatched_tracks_first:
            if i not in matched_track_indices:
                s = strack_pool[i]
                if s.state == TrackState.Lost:
                    pass  # already lost, time_since_update incremented in predict

        # ---- Step 5: Create new tracks from unmatched high-conf detections ----
        for d_idx in unmatched_dets_first:
            det_orig_idx = int(high_indices[d_idx])
            score = scores[det_orig_idx]
            new_track = STrack(bboxes[det_orig_idx], score)
            new_track.activate(self.frame_id)
            self.tracked_stracks.append(new_track)
            results.append((det_orig_idx, new_track.track_id))

        # ---- Step 6: Remove tracks that exceeded max_time_lost ----
        new_tracked = []
        new_lost = []
        for strack in self.tracked_stracks:
            if strack.state == TrackState.Tracked:
                new_tracked.append(strack)
            elif strack.state == TrackState.Lost:
                new_lost.append(strack)
            else:
                self.removed_stracks.append(strack)

        for strack in self.lost_stracks:
            if strack.state == TrackState.Tracked:
                new_tracked.append(strack)
            elif strack.time_since_update > self.max_time_lost:
                strack.mark_removed()
                self.removed_stracks.append(strack)
            else:
                new_lost.append(strack)

        self.tracked_stracks = new_tracked
        self.lost_stracks = new_lost

        # Limit removed_stracks to prevent unbounded memory growth
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-500:]

        # Sort results by detection index for stable output
        results.sort(key=lambda x: x[0])
        return results

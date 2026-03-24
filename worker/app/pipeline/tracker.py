"""IoU + centroid tracker with size gating, velocity prediction, appearance matching,
and graveyard re-identification for robust occlusion recovery."""
import logging

import numpy as np

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
        self.next_id = 0
        self.active_tracks: dict[int, tuple] = {}  # track_id -> last bbox
        self.missing_count: dict[int, int] = {}  # track_id -> frames since last seen
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
        self._frame_counter = 0

    def seed(self, known_bboxes: dict[int, tuple], histograms: dict[int, list[float]] | None = None,
             group_ids: set[int] | None = None):
        """Seed the tracker with known track_id -> bbox mappings from a prior detection pass."""
        for tid, bbox in known_bboxes.items():
            self.active_tracks[tid] = bbox
            self.missing_count[tid] = 0
            self.track_avg_area[tid] = _bbox_area(bbox)
            self.track_velocity[tid] = (0.0, 0.0)
        if histograms:
            for tid, hist in histograms.items():
                self.track_histogram[tid] = hist
        if group_ids:
            self.group_ids = set(group_ids)
        else:
            self.group_ids = set(known_bboxes.keys())
        # Initialize pairwise distances for the group
        self._update_group_distances()
        self.next_id = max(known_bboxes.keys()) + 1 if known_bboxes else 0

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

    def update(self, bboxes: list[tuple], histograms: list[list[float]] | None = None) -> list[int]:
        """Assign track IDs to a list of bounding boxes for one frame.

        Args:
            bboxes: List of (x_min, y_min, x_max, y_max) tuples.
            histograms: Optional parallel list of color histograms for appearance matching.

        Returns a list of track_ids parallel to the input bboxes.
        """
        hists = histograms or [None] * len(bboxes)

        if not self.active_tracks:
            # First frame: assign new IDs
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
                best_dist = self.max_centroid_dist
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
        still_unmatched = []
        for det_idx in unmatched_dets:
            best_grave_tid = None
            best_score = -1
            det_area = _bbox_area(bboxes[det_idx])
            for g_tid, g_bbox in self.graveyard.items():
                g_avg = self.graveyard_area.get(g_tid, 0)
                if g_avg <= 0 or det_area <= 0:
                    continue
                ratio = det_area / g_avg if det_area >= g_avg else g_avg / det_area
                if ratio > self.max_size_ratio:
                    continue
                # Use predicted graveyard position (extrapolate velocity)
                g_vel = self.graveyard_velocity.get(g_tid, (0.0, 0.0))
                g_age = self.graveyard_age.get(g_tid, 0) + self.max_missing
                predicted_g = _shift_bbox(g_bbox, g_vel[0] * g_age, g_vel[1] * g_age)
                dist = centroid_distance(bboxes[det_idx], predicted_g)
                if dist > self.max_centroid_dist * 2.0:
                    continue
                # Score: proximity (inverted distance) + appearance
                prox = max(0, 1.0 - dist / (self.max_centroid_dist * 2.0))
                g_hist = self.graveyard_histogram.get(g_tid)
                app = self._appearance_similarity(hists[det_idx], g_tid) if not g_hist else (
                    float(sum(min(a, b) for a, b in zip(hists[det_idx], g_hist))) if hists[det_idx] and g_hist and len(hists[det_idx]) == len(g_hist) else 0.5
                )
                score = prox * 0.5 + app * 0.5
                if score > best_score:
                    best_score = score
                    best_grave_tid = g_tid

            if best_grave_tid is not None and best_score > 0.3:
                assignments[det_idx] = best_grave_tid
                self.track_avg_area[best_grave_tid] = self.graveyard_area.pop(best_grave_tid)
                self.track_velocity[best_grave_tid] = self.graveyard_velocity.pop(best_grave_tid, (0.0, 0.0))
                if best_grave_tid in self.graveyard_histogram:
                    self.track_histogram[best_grave_tid] = self.graveyard_histogram.pop(best_grave_tid)
                del self.graveyard[best_grave_tid]
                del self.graveyard_age[best_grave_tid]
                logger.info(f"Re-identified track {best_grave_tid} after occlusion (score={best_score:.2f})")
            else:
                still_unmatched.append(det_idx)

        # Appearance-only re-lock: for group members in the graveyard that weren't
        # recovered by position, try matching purely on appearance + size with
        # 3-frame confirmation (inspired by fencing analyzer's re-lock pattern).
        relock_recovered = []
        unrecovered_group_graves = [
            g_tid for g_tid in self.graveyard
            if g_tid in self.group_ids and g_tid not in {assignments[i] for i in range(len(assignments)) if assignments[i] is not None}
        ]
        if still_unmatched and unrecovered_group_graves:
            for det_idx in still_unmatched:
                if not hists[det_idx]:
                    continue
                det_area = _bbox_area(bboxes[det_idx])
                best_g_tid = None
                best_app = -1
                for g_tid in unrecovered_group_graves:
                    if not self._size_compatible(bboxes[det_idx], g_tid):
                        # Check against graveyard area directly
                        g_avg = self.graveyard_area.get(g_tid, 0)
                        if g_avg <= 0 or det_area <= 0:
                            continue
                        ratio = det_area / g_avg if det_area >= g_avg else g_avg / det_area
                        if ratio > self.max_size_ratio:
                            continue
                    g_hist = self.graveyard_histogram.get(g_tid)
                    if not g_hist or not hists[det_idx] or len(g_hist) != len(hists[det_idx]):
                        continue
                    app = float(sum(min(a, b) for a, b in zip(hists[det_idx], g_hist)))
                    if app > best_app:
                        best_app = app
                        best_g_tid = g_tid

                if best_g_tid is not None and best_app > 0.6:
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
                        del self.graveyard[best_g_tid]
                        del self.graveyard_age[best_g_tid]
                        self.graveyard_velocity.pop(best_g_tid, None)
                        del self.relock_candidates[best_g_tid]
                        unrecovered_group_graves.remove(best_g_tid)
                        relock_recovered.append(det_idx)
                        logger.info(f"Re-locked track {best_g_tid} via appearance (score={best_app:.2f})")

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
            self._update_avg_area(tid, bboxes[det_idx])
            self._update_histogram(tid, hists[det_idx])
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
                # Move to graveyard
                self.graveyard[tid] = self.active_tracks[tid]
                self.graveyard_area[tid] = self.track_avg_area.get(tid, 0)
                self.graveyard_age[tid] = 0
                self.graveyard_velocity[tid] = self.track_velocity.get(tid, (0.0, 0.0))
                if tid in self.track_histogram:
                    self.graveyard_histogram[tid] = self.track_histogram[tid]
                logger.debug(f"Track {tid} moved to graveyard after {self.max_missing} missing frames")

        # Age out graveyard entries
        expired_graves = []
        for g_tid in self.graveyard:
            self.graveyard_age[g_tid] = self.graveyard_age.get(g_tid, 0) + 1
            if self.graveyard_age[g_tid] > self.graveyard_frames:
                expired_graves.append(g_tid)
        for g_tid in expired_graves:
            del self.graveyard[g_tid]
            del self.graveyard_area[g_tid]
            del self.graveyard_age[g_tid]
            self.graveyard_velocity.pop(g_tid, None)
            self.graveyard_histogram.pop(g_tid, None)

        self.active_tracks = new_tracks
        self.missing_count = new_missing
        self._frame_counter += 1

        # Clean stale relock candidates (not seen for >5 frames)
        stale = [tid for tid, c in self.relock_candidates.items()
                 if self._frame_counter - c["frame_idx"] > 5]
        for tid in stale:
            del self.relock_candidates[tid]

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

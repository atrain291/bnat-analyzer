"""Simple IoU + centroid tracker for assigning stable IDs across frames."""
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


def centroid_distance(box_a: tuple, box_b: tuple) -> float:
    """Euclidean distance between centroids of two boxes."""
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


class SimpleTracker:
    """Track persons across frames using IoU with centroid and positional fallback.

    Handles camera movement by:
    1. Keeping unmatched tracks alive for `max_missing` frames
    2. Using spatial rank order (left-to-right) as final fallback when
       both IoU and centroid distance fail (camera pan/shift)
    """

    def __init__(self, iou_threshold: float = 0.2, max_centroid_dist: float = 0.3, max_missing: int = 30):
        self.iou_threshold = iou_threshold
        self.max_centroid_dist = max_centroid_dist
        self.max_missing = max_missing
        self.next_id = 0
        self.active_tracks: dict[int, tuple] = {}  # track_id -> last bbox
        self.missing_count: dict[int, int] = {}  # track_id -> frames since last seen

    def seed(self, known_bboxes: dict[int, tuple]):
        """Seed the tracker with known track_id -> bbox mappings from a prior detection pass.

        This ensures the first frame's detections are matched to the correct IDs.
        """
        for tid, bbox in known_bboxes.items():
            self.active_tracks[tid] = bbox
            self.missing_count[tid] = 0
        self.next_id = max(known_bboxes.keys()) + 1 if known_bboxes else 0

    def update(self, bboxes: list[tuple]) -> list[int]:
        """Assign track IDs to a list of bounding boxes for one frame.

        Returns a list of track_ids parallel to the input bboxes.
        """
        if not self.active_tracks:
            # First frame: assign new IDs
            ids = []
            for bbox in bboxes:
                ids.append(self.next_id)
                self.active_tracks[self.next_id] = bbox
                self.missing_count[self.next_id] = 0
                self.next_id += 1
            return ids

        # Greedy matching: for each detection, find best matching track
        track_ids = list(self.active_tracks.keys())
        track_bboxes = [self.active_tracks[tid] for tid in track_ids]
        used_tracks = set()
        assignments = [None] * len(bboxes)

        # Sort detections by area (largest first for priority)
        det_order = sorted(range(len(bboxes)), key=lambda i: -(
            (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])
        ))

        for det_idx in det_order:
            best_score = -1
            best_tid_idx = -1

            for tid_idx, tid in enumerate(track_ids):
                if tid in used_tracks:
                    continue
                iou = compute_iou(bboxes[det_idx], track_bboxes[tid_idx])
                if iou >= self.iou_threshold and iou > best_score:
                    best_score = iou
                    best_tid_idx = tid_idx

            # Fallback to centroid distance if no IoU match
            if best_tid_idx == -1:
                best_dist = self.max_centroid_dist
                for tid_idx, tid in enumerate(track_ids):
                    if tid in used_tracks:
                        continue
                    dist = centroid_distance(bboxes[det_idx], track_bboxes[tid_idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_tid_idx = tid_idx

            if best_tid_idx >= 0:
                assignments[det_idx] = track_ids[best_tid_idx]
                used_tracks.add(track_ids[best_tid_idx])

        # Positional fallback: if same number of unmatched detections and tracks,
        # match by left-to-right order (handles camera movement where everything shifts)
        unmatched_dets = [i for i in range(len(bboxes)) if assignments[i] is None]
        unmatched_tids = [tid for tid in track_ids if tid not in used_tracks]

        if unmatched_dets and unmatched_tids and len(unmatched_dets) == len(unmatched_tids):
            # Sort both by x-centroid position
            unmatched_dets.sort(key=lambda i: (bboxes[i][0] + bboxes[i][2]) / 2)
            unmatched_tids.sort(key=lambda tid: (self.active_tracks[tid][0] + self.active_tracks[tid][2]) / 2)
            for det_idx, tid in zip(unmatched_dets, unmatched_tids):
                assignments[det_idx] = tid
                used_tracks.add(tid)
            unmatched_dets = []
            unmatched_tids = [tid for tid in track_ids if tid not in used_tracks]

        # Assign new IDs to any remaining unmatched detections
        for det_idx in range(len(bboxes)):
            if assignments[det_idx] is None:
                assignments[det_idx] = self.next_id
                self.next_id += 1

        # Update active tracks: matched tracks get new bbox, unmatched tracks age out
        new_tracks = {}
        new_missing = {}
        for det_idx, tid in enumerate(assignments):
            new_tracks[tid] = bboxes[det_idx]
            new_missing[tid] = 0

        # Keep unmatched tracks alive for max_missing frames
        for tid in unmatched_tids:
            age = self.missing_count.get(tid, 0) + 1
            if age <= self.max_missing:
                new_tracks[tid] = self.active_tracks[tid]
                new_missing[tid] = age

        self.active_tracks = new_tracks
        self.missing_count = new_missing

        return assignments


def run_tracker(
    all_frames_detections: list[list[dict]],
    min_frame_ratio: float = 0.3,
) -> list[dict]:
    """Run tracker across detection frames and return stable person summaries.

    Args:
        all_frames_detections: List of frames, each frame is a list of person dicts
            with 'bbox' and 'dancer_pose' keys.
        min_frame_ratio: Minimum fraction of frames a person must appear in to be kept.

    Returns:
        List of detected person dicts: {track_id, bbox, representative_pose, frame_count, area}
    """
    tracker = SimpleTracker()
    # track_id -> list of (frame_idx, bbox, area, pose_data)
    track_history: dict[int, list] = {}

    for frame_idx, detections in enumerate(all_frames_detections):
        bboxes = [d["bbox"] for d in detections]
        if not bboxes:
            continue

        track_ids = tracker.update(bboxes)

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

        results.append({
            "track_id": tid,
            "bbox": median_bbox,
            "representative_pose": best_entry[3].get("dancer_pose", {}),
            "frame_count": len(history),
            "area": float(np.mean([h[2] for h in history])),
        })

    # Sort by area descending (largest person first)
    results.sort(key=lambda r: -r["area"])
    logger.info(f"Tracker found {len(results)} stable persons from {len(track_history)} raw tracks")
    return results

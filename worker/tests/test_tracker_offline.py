"""Test tracker against dumped detections — no GPU needed, runs in seconds.

Run inside the worker container:
    python -m tests.test_tracker_offline /app/uploads/detections_dump.json --tracks 0,1,3

Prints per-frame tracking results: which selected tracks were captured, which were lost.
"""
import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_path")
    parser.add_argument("--tracks", required=True, help="Comma-separated track IDs to select (e.g., 0,1,3)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0=all)")
    args = parser.parse_args()

    selected_ids = set(int(t) for t in args.tracks.split(","))

    with open(args.dump_path) as f:
        dump = json.load(f)

    frames = dump["frames"]
    if args.max_frames > 0:
        frames = frames[:args.max_frames]

    logger.info(f"Loaded {len(frames)} frames, selecting tracks {selected_ids}")

    # Run detection pass tracker to get stable IDs
    from app.pipeline.tracker import SimpleTracker, run_tracker

    # First: run detection-pass tracker on first 50 frames to get seed bboxes
    det_frames = []
    for f in frames[:50]:
        det_frames.append([{
            "bbox": tuple(p["bbox"]),
            "dancer_pose": p.get("dancer_pose", {}),
            "color_histogram": p.get("color_histogram"),
        } for p in f["persons"]])

    persons = run_tracker(det_frames, min_frame_ratio=0.2)
    logger.info(f"Detection pass found {len(persons)} stable tracks")
    for p in persons:
        logger.info(f"  Track {p['track_id']}: {p['frame_count']} frames, area={p['area']:.4f}")

    # Build seed data
    seed_bboxes = {}
    seed_histograms = {}
    for p in persons:
        tid = p["track_id"]
        b = p["bbox"]
        seed_bboxes[tid] = (b["x_min"], b["y_min"], b["x_max"], b["y_max"])
        if "color_histogram" in p:
            seed_histograms[tid] = p["color_histogram"]

    if not all(t in seed_bboxes for t in selected_ids):
        missing = selected_ids - set(seed_bboxes.keys())
        logger.error(f"Selected tracks {missing} not found in detection pass. Available: {set(seed_bboxes.keys())}")
        sys.exit(1)

    # Now run full tracker
    metadata = dump["metadata"]
    fps = metadata["fps"]
    from app.pipeline.pose_config import POSE_FRAME_SKIP
    effective_fps = fps / POSE_FRAME_SKIP
    tracker = SimpleTracker(effective_fps=effective_fps)
    tracker.seed(seed_bboxes, histograms=seed_histograms, group_ids=selected_ids)

    frame_counts = {tid: 0 for tid in selected_ids}
    last_captured = {tid: -1 for tid in selected_ids}
    total_captured = 0
    gap_lengths = {tid: [] for tid in selected_ids}
    current_gap = {tid: 0 for tid in selected_ids}

    for fi, frame in enumerate(frames):
        bboxes = [tuple(p["bbox"]) for p in frame["persons"]]
        hists = [p.get("color_histogram") for p in frame["persons"]]
        has_hists = any(h for h in hists)

        track_ids = tracker.update(bboxes, histograms=hists if has_hists else None)

        captured_this = set()
        for det_idx, tid in enumerate(track_ids):
            if tid in selected_ids:
                captured_this.add(tid)
                frame_counts[tid] += 1
                total_captured += 1
                if current_gap[tid] > 0:
                    gap_lengths[tid].append(current_gap[tid])
                    current_gap[tid] = 0
                last_captured[tid] = fi

        for tid in selected_ids:
            if tid not in captured_this:
                current_gap[tid] += 1

        # Log every 100 frames
        if fi % 100 == 0 or fi == len(frames) - 1:
            active_group = [t for t in tracker.active_tracks if t in selected_ids]
            grave_group = [t for t in tracker.graveyard if t in selected_ids]
            ts_ms = frame.get("timestamp_ms", 0)
            logger.info(f"Frame {fi:4d} ({ts_ms/1000:.1f}s): captured={sorted(captured_this)}, "
                        f"active_group={active_group}, grave_group={grave_group}, "
                        f"total_active={len(tracker.active_tracks)}, n_dets={len(bboxes)}")

    logger.info("=" * 60)
    logger.info("RESULTS:")
    for tid in sorted(selected_ids):
        pct = frame_counts[tid] / len(frames) * 100
        gaps = gap_lengths[tid]
        max_gap = max(gaps) if gaps else 0
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        logger.info(f"  Track {tid}: {frame_counts[tid]}/{len(frames)} frames ({pct:.1f}%), "
                    f"gaps={len(gaps)}, max_gap={max_gap}, avg_gap={avg_gap:.0f}")
    total_pct = total_captured / (len(frames) * len(selected_ids)) * 100
    logger.info(f"  Overall: {total_pct:.1f}% capture rate")


if __name__ == "__main__":
    main()

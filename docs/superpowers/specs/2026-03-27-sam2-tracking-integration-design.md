# SAM 2 Tracking Integration — Design Spec

## Problem

The current detection/tracking pipeline (RTMPose detection + ByteTrack/custom tracker) fundamentally cannot reliably track dancers through a video. RTMPose detects "person-shaped blobs" each frame independently — it has no concept of identity. The tracker tries to match bboxes frame-to-frame but fails when dancers overlap, merge, or change formation. Sitting audience members get detected as dancers. Skeletons swap between people from one frame to the next.

After multiple iterations (custom tracker, ByteTrack, identity recovery, biometrics, Re-ID CNN, motion correlation), the core issue remains: **top-down pose estimation does not track people.**

## Solution

Replace bbox-based tracking with **SAM 2.1 (Segment Anything Model 2.1)** pixel-level video segmentation. The user clicks on dancers in a video frame, and SAM 2 tracks those exact people at the pixel level through the entire video. RTMPose then runs on pre-cropped individual dancer images — guaranteed one skeleton per crop, no confusion.

**Model:** `sam2.1_hiera_base_plus.pt` (SAM 2.1 Base+, ~4GB VRAM, ~28fps on 1080p)
**Package:** `sam2` PyPI package (Meta's official release)

## User Flow

1. Upload video → backend saves file, dispatches async transcode if needed
2. Once transcode complete (status `uploaded`), frontend shows video player with scrub bar on the **Select Dancers** page
3. User scrubs to a frame where dancers are clearly visible
4. User clicks on each dancer they want tracked (colored dots appear at click points)
5. User optionally labels each dancer ("Somia", "Surabhi", "Maya")
6. User clicks "Analyze from here" — sends frame timestamp + click coordinates to backend
7. Processing starts from the selected frame forward (content before that frame is not analyzed)
8. Status page shows: "Tracking dancers..." → "Estimating poses..." → "Analyzing..." → "Complete"
9. Review page shows skeleton overlays, scores, coaching (same as current)

**Single-dancer shortcut:** If user clicks on only 1 dancer, the pipeline runs in single-dancer mode (no multi-dancer routing). The click-to-select step is still required — it replaces the old auto-detect flow for all cases.

## Architecture

### Containers

| Container | Role | GPU Usage | Volumes | Celery concurrency |
|-----------|------|-----------|---------|-------------------|
| `api` | FastAPI backend, serves video files | None | uploads | N/A |
| `worker` | Celery: transcode, RTMPose, scoring, LLM coaching | RTMPose ~500MB | uploads | 1 |
| `sam2_worker` | Celery on `sam2_tracking` queue: SAM 2 segmentation | SAM 2 Base+ ~4GB | uploads | 1 |
| `wham_worker` | Celery on `wham_3d` queue: WHAM 3D (unchanged) | HMR2 ~2.9GB + WHAM ~0.2GB | — (DB only) | 1 |
| `postgres` | Database (unchanged) | None | pgdata | N/A |
| `redis` | Message broker (unchanged) | None | redisdata | N/A |
| `frontend` | React app (unchanged structure) | None | — | N/A |

All GPU workers run with `concurrency=1` and `prefetch-multiplier=1` to prevent GPU contention. SAM 2 and RTMPose never run simultaneously for the same performance, but concurrent performances could overlap — the concurrency=1 setting on each worker ensures only one GPU task runs per container at a time.

The `sam2_worker` mounts the same `uploads` volume at `/app/uploads` as the main `worker`, so video paths passed via Celery (`/app/uploads/<key>`) resolve identically in both containers.

### Data Flow

```
Upload Video
    |
    v
Backend: save file, dispatch transcode task to worker queue
Status: "transcoding"
    |
    v
Worker: ensure_browser_playable() (GPU transcode HEVC->H.264 if needed)
On completion, set status: "uploaded"
    |
    v
Frontend: /select-frame/:performanceId
  - Video player with scrub bar (video now browser-playable)
  - User scrubs to good frame, clicks on dancers
  - Sends: {start_timestamp_ms, prompts: [{x, y, label}...]}
    |
    v
Backend: store click prompts, create PerformanceDancer records
Dispatch SAM 2 task to sam2_tracking queue
Status: "tracking"
    |
    v
SAM 2 Worker:
  1. Open video file via SAM2VideoPredictor.init_state(video_path=...)
  2. Compute start_frame_idx from start_timestamp_ms and fps
  3. Add click prompts as point prompts at start_frame_idx
     (normalized 0-1 coords -> pixel coords: x * width, y * height)
  4. Call predictor.propagate_in_video() forward from start_frame_idx
  5. For each output frame, for each dancer:
     - Derive bbox from mask (min/max of nonzero pixels), normalize to 0-1
     - Record mask IoU confidence
     - Store to tracking_frames table (batch inserts every 500 rows)
  6. Check Redis cancel flag every 50 frames
  7. On completion: dispatch pose pipeline via send_task()
Status: "tracking" -> dispatches pose pipeline
    |
    v
Main Worker:
  1. Read tracking_frames from DB per dancer (ordered by timestamp_ms)
  2. Stream video via ffmpeg using output-mode seeking (-i video -ss start_time)
     for frame-accurate seeking from start_timestamp_ms
  3. For each frame, for each dancer:
     - Look up tracking bbox for this timestamp (exact match by timestamp_ms)
     - Skip if mask_iou < 0.3 or bbox area is zero (dancer off-screen)
     - Crop frame to bbox + 25% padding (denormalize bbox to pixels first)
     - Run RTMPose on crop -> single-person keypoints
     - Transform keypoints to full-frame normalized coordinates
  4. Store frames, compute angles, scores
  5. LLM coaching per dancer
  6. Dispatch WHAM 3D (fire-and-forget)
  Status: "processing" -> "complete"
    |
    v
WHAM Worker (unchanged, fire-and-forget)
```

### Performance Status Lifecycle

**Current:** `queued -> detecting -> awaiting_selection -> processing -> complete | failed`

**New:** `transcoding -> uploaded -> tracking -> processing -> complete | failed`

- `transcoding`: video being transcoded for browser playback (instant for H.264 input)
- `uploaded`: video ready, waiting for user to select dancers
- `tracking`: SAM 2 segmenting video
- `processing`: RTMPose + analysis + coaching
- `complete` / `failed`: same as before

### Database Changes

**New table: `tracking_frames`**

```
tracking_frames:
  id                    INTEGER PRIMARY KEY
  performance_id        INTEGER FK -> performances (ON DELETE CASCADE)
  dancer_index          INTEGER (0, 1, 2... — order of user clicks)
  timestamp_ms          INTEGER
  bbox                  JSON ({x_min, y_min, x_max, y_max} normalized 0-1)
  mask_iou              FLOAT (SAM 2 confidence score)
  INDEX (performance_id, dancer_index, timestamp_ms)
```

Expected size: ~4300 frames x 3 dancers = ~13,000 rows per performance. Cleaned up automatically via ON DELETE CASCADE when performance is deleted.

**Performance model additions:**

```
start_timestamp_ms    INTEGER NULLABLE (frame where user selected dancers)
click_prompts         JSON NULLABLE ([{x, y, label}, ...] normalized 0-1)
```

**PerformanceDancer model:**

`track_id` column set to `dancer_index` (0, 1, 2...) for new performances. Column kept for backwards compat.

**DetectedPerson table:** No longer populated by the new flow. Table kept for backwards compat.

**DB migration must also:**
- Set any performance with status `queued`, `detecting`, or `awaiting_selection` to `failed` with error "Pipeline upgraded — please re-upload this video"
- Add `tracking_frames` table with ON DELETE CASCADE at both SQL and ORM levels
- Add `start_timestamp_ms` and `click_prompts` nullable columns to `performances`

**Both backend AND worker need the `TrackingFrame` ORM model** — the backend needs it for the `reset-tracking` endpoint to delete rows. The SAM 2 worker needs it for inserts. The main worker needs it for reads.

### SAM 2 Video Predictor API

SAM 2's `SAM2VideoPredictor` does NOT accept streaming frames via an ffmpeg pipe. It requires either:
- A directory of pre-extracted JPEG/PNG frames, or
- A video file path (SAM 2.1 supports mp4 natively)

**We use the video file path approach** — simplest, no temp files:

```python
predictor = build_sam2_video_predictor(model_cfg, checkpoint_path)
state = predictor.init_state(video_path="/app/uploads/video.mp4")

# Add click prompts at the user's selected frame
# Points must be in ORIGINAL frame pixel coords (SAM 2 handles internal rescaling)
frame_idx = int(start_timestamp_ms * fps / 1000)
for i, prompt in enumerate(click_prompts):
    points = np.array([[prompt["x"] * width, prompt["y"] * height]])
    labels = np.array([1])  # 1 = positive (foreground)
    predictor.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=i,
                                     points=points, labels=labels)

# Propagate forward
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    for obj_id, mask in zip(obj_ids, masks):
        # mask is a binary tensor (H, W)
        # Derive bbox from mask extent
        ...
```

**Frame skip handling:** SAM 2 processes all frames internally (its propagation model needs frame-to-frame continuity). We store tracking results only for frames at our desired skip interval (every `SAM2_FRAME_SKIP` frames). This decouples SAM 2's internal processing from our storage/pose cadence.

### SAM 2 Memory Management

SAM 2's video predictor maintains a memory bank of past frame features that grows over time. For long videos (289s at ~30fps = ~8600 frames), this can become substantial. Mitigations:

1. **Configure `max_cond_frames_in_attn`** (default 6): limits attention window. Keep at default.
2. **Process in temporal chunks** if GPU memory exceeds limits: 1000-frame chunks with 50-frame overlap, re-seeding from last good mask at chunk boundaries.
3. **Monitor GPU memory** during inference; log warnings at 80% VRAM usage.

For Base+ on 12GB GPU processing ~8600 frames, peak usage ~6-7GB — within budget. Chunking is the fallback if this estimate proves tight.

### New SAM 2 Worker Container

**Directory:** `sam2_worker/`

**Structure:**
```
sam2_worker/
  Dockerfile           # pytorch:2.5.1-cuda12.4, install sam2
  requirements.txt     # sam2>=1.0, celery[redis], sqlalchemy, psycopg2-binary, opencv-python-headless
  app/
    __init__.py
    celery_app.py      # Celery app on sam2_tracking queue, concurrency=1, prefetch=1
    db.py              # Postgres session (same pattern as wham_worker)
    models.py          # Minimal ORM: Performance, PerformanceDancer, TrackingFrame
    tasks.py           # run_sam2_tracking task (error handling, cancel check, dispatch)
    inference.py       # SAM 2 video predictor logic
```

**Error handling in tasks.py:**
- Wrap inference in try/except; on failure set performance status to `failed` with error message
- Check Redis cancel flag every 50 frames; on cancel, store partial tracking data, dispatch pose pipeline on what exists
- If SAM 2 model fails to load, set status `failed` immediately

**Model files:** `$HOME/bharatanatyam-data/sam2/sam2.1_hiera_base_plus.pt` (mounted volume).

### Handoff: SAM 2 → Main Worker

SAM 2 worker dispatches pose pipeline via Celery `send_task()`:

```python
selected_tracks = [
    {"track_id": pd.track_id, "performance_dancer_id": pd.id, "label": pd.label}
    for pd in performance_dancers
]
_celery_sender.send_task(
    "worker.app.tasks.video_pipeline.run_pipeline",
    args=[performance_id, video_path, selected_tracks],
    queue="video_pipeline",
)
```

Schema for `selected_tracks` items: `{"track_id": int (dancer_index), "performance_dancer_id": int, "label": str|null}`. This matches the existing `run_pipeline` signature.

### Video Seeking — Frame Accuracy

ffmpeg input-mode seeking (`-ss` before `-i`) seeks to the nearest **keyframe**, which can be seconds off. For frame-accurate seeking:

**SAM 2 worker:** Not an issue — SAM 2's `init_state(video_path=...)` handles its own frame decoding. We specify the `frame_idx` for prompts and propagation start. The frame index is computed as `int(start_timestamp_ms / 1000 * fps)`.

**Main worker (RTMPose cropped):** Use **output-mode seeking** for frame accuracy:
```
ffmpeg -i video.mp4 -ss <start_time> -vf "select=..." -f rawvideo -pix_fmt bgr24 pipe:1
```
With `-ss` AFTER `-i`, ffmpeg decodes from the nearest keyframe and discards frames until the target time. Slower but frame-accurate. For a 289s video starting at 30s, the overhead is decoding ~2-5s of discarded frames — negligible.

Timestamp calculation for each output frame:
```
timestamp_ms = start_timestamp_ms + (frame_idx * frame_skip / fps * 1000)
```

### Frontend Changes

**New page: `/select-frame/:performanceId`** (replaces DancerSelection.tsx)

- Video player (`<video>` element) with native scrub bar
- Canvas overlay for click interaction (positioned over video, matching video dimensions)
- Click coordinates normalized to video's **natural dimensions** (not CSS display size):
  `norm_x = (click_x - video_rect.left) / video_rect.width * (video.videoWidth / video_rect.width)`
  Simplified: `norm_x = offsetX / displayWidth` (since video maintains aspect ratio)
- Show list of selected dancers below video (color dot + label + remove button)
- "Analyze from here" button → POST `/api/performances/{id}/select-frame`
- Minimum 1 dancer selected to proceed
- Disable interactions if status is not `uploaded` (show "Preparing video..." during transcode)

**Modified: Dashboard.tsx**

- Upload navigates to `/select-frame/:performanceId`
- Status `transcoding` shows spinner + "Preparing video..."
- Status `uploaded` shows "Select dancers" link
- Status `tracking` shows spinner + "Tracking dancers..."

**Modified: ProcessingStatus.tsx**

- Handle `tracking` stage with progress bar
- Redirect to `/select-frame/:id` when status is `uploaded` (same pattern as current `awaiting_selection` → `/select-dancers/:id`)
- Show "Stop" button for `tracking` status (not just `processing`)

**Removed: DancerSelection.tsx**

**Modified: App.tsx**

- Replace `/select-dancers/:performanceId` route with `/select-frame/:performanceId`

### Backend API Changes

**Modified: POST /api/upload/**

- Saves file, creates Performance with status `transcoding`
- Dispatches transcode task to worker
- Returns `UploadResponse` (task_id may be null or the transcode task_id)

**New: POST /api/performances/{id}/select-frame**

```
Request body:
{
  "start_timestamp_ms": 30000,
  "prompts": [
    {"x": 0.15, "y": 0.45, "label": "Somia"},
    {"x": 0.35, "y": 0.42, "label": "Maya"},
    {"x": 0.55, "y": 0.44, "label": "Surabhi"}
  ]
}

Response:
{
  "status": "tracking",
  "dancers_selected": 3
}
```

- Validates performance status is `uploaded`
- Stores `start_timestamp_ms` and `click_prompts` on Performance
- Creates PerformanceDancer records (track_id = dancer_index = 0, 1, 2...)
- Dispatches SAM 2 tracking task to `sam2_tracking` queue
- Updates status to `tracking`

**New: POST /api/performances/{id}/reset-tracking**

- Only callable when status is `uploaded`, `tracking`, or `failed`
- If status is `tracking`: sets Redis cancel flag first, then proceeds
- Deletes `tracking_frames`, `frames`, `analyses` rows for this performance
- Deletes `PerformanceDancer` records
- Clears `start_timestamp_ms` and `click_prompts`
- Sets status to `uploaded`

**Modified: POST /api/performances/{id}/stop**

- Add `tracking` and `transcoding` to stoppable statuses

**Modified: GET /api/performances/{id}/status**

- Handle `transcoding` and `tracking` stages in progress reporting

### Main Worker Pose Pipeline Changes

**Modified: `worker/app/tasks/video_pipeline.py`**

The multi-dancer branch changes from:
1. Read seed bboxes from DetectedPerson → seed tracker → `run_pose_estimation_multi()`

To:
1. Read per-dancer bboxes from `tracking_frames` table (ordered by dancer_index, timestamp_ms)
2. For each dancer, call `run_pose_estimation_cropped()`

The `selected_tracks` list still uses the same schema. The per-dancer routing, frame storage, angle computation, scoring, and coaching are unchanged.

**Duration adjustment:** `metadata["duration_ms"]` is adjusted to reflect only the analyzed portion:
```
effective_duration_ms = metadata["duration_ms"] - start_timestamp_ms
```
This ensures scores and coaching reference the correct time span.

**Coverage check:** Uses the tracking data extent, not full video duration. If the tracking was cancelled early, coverage is measured against the tracking_frames that exist.

**New: `worker/app/tasks/transcode.py`**

Small Celery task: runs `ensure_browser_playable()`, sets status to `uploaded`.

**New function in `worker/app/pipeline/pose.py`:**

`run_pose_estimation_cropped(video_path, metadata, dancer_bboxes, start_ms, frame_skip, ...)`

- `dancer_bboxes`: list of `{timestamp_ms, bbox, mask_iou}` from tracking_frames
- Uses output-mode ffmpeg seeking for frame accuracy
- For each frame at the target cadence:
  - Look up tracking bbox by exact `timestamp_ms` match
  - Skip if `mask_iou < 0.3` or bbox area is zero
  - Denormalize bbox to pixels, add 25% padding, clamp to frame bounds
  - Crop the frame
  - Run RTMPose on crop (may resize if `POSE_MAX_HEIGHT` is set)
  - RTMPose outputs keypoints in the **model input resolution**
  - Scale keypoints back to crop pixel coords: `kp_crop_px = kp_model * (crop_px / model_input_px)`
  - Transform to full-frame normalized coords: `full_norm = kp_crop_px / frame_px + crop_offset / frame_px`
- Yields frame dicts (same format as current pipeline)

**Coordinate transform, worked example:**

Given: frame 1920x1080, bbox normalized (0.2, 0.1, 0.5, 0.9), 25% padding, POSE_MAX_HEIGHT=720

1. Denormalize bbox: (384, 108, 960, 972) → size 576x864
2. Add 25% pad: (240, 0, 1104, 1080) → crop 864x1080 (clamped)
3. Crop pixels, resize to max_height=720 → model input 576x720
4. RTMPose detects nose at model pixel (288, 200)
5. Scale to crop pixels: (288 * 864/576, 200 * 1080/720) = (432, 300)
6. Full-frame pixel: (240 + 432, 0 + 300) = (672, 300)
7. Normalize: (672/1920, 300/1080) = (0.35, 0.278)

### Files Removed

| File | Lines | Reason |
|------|-------|--------|
| `worker/app/pipeline/tracker.py` | 1206 | Replaced by SAM 2 |
| `worker/app/pipeline/bytetrack.py` | 440 | Replaced by SAM 2 |
| `worker/app/pipeline/reid.py` | ~130 | No longer needed |
| `worker/app/pipeline/appearance.py` | ~156 | No longer needed for tracking |
| `worker/export_reid_model.py` | ~40 | No longer needed |
| `worker/app/tasks/detect_dancers.py` | ~110 | No more detection pass |
| `frontend/src/pages/DancerSelection.tsx` | ~330 | Replaced by SelectFrame |
| **Total removed** | **~2412** | |

Note: `biometrics.py` kept (used by WHAM worker). `scipy` kept in requirements (librosa dependency).

### Files Added

| File | Purpose |
|------|---------|
| `sam2_worker/Dockerfile` | SAM 2 container (pytorch:2.5.1-cuda12.4, sam2 package) |
| `sam2_worker/requirements.txt` | sam2>=1.0, celery[redis], sqlalchemy, psycopg2-binary, opencv-python-headless |
| `sam2_worker/app/__init__.py` | Package init |
| `sam2_worker/app/celery_app.py` | Celery on sam2_tracking queue, concurrency=1 |
| `sam2_worker/app/db.py` | Postgres session |
| `sam2_worker/app/models.py` | ORM: Performance, PerformanceDancer, TrackingFrame |
| `sam2_worker/app/tasks.py` | run_sam2_tracking task |
| `sam2_worker/app/inference.py` | SAM 2 video prediction |
| `worker/app/tasks/transcode.py` | Async transcode task |
| `frontend/src/pages/SelectFrame.tsx` | Click-to-select UI |
| `alembic/versions/xxx_add_tracking.py` | DB migration |

### Files Modified

| File | Change |
|------|--------|
| `docker-compose.yml` | Add sam2_worker service (GPU, uploads volume, concurrency=1) |
| `backend/app/api/routes/upload.py` | Dispatch transcode (not detection), status "transcoding" |
| `backend/app/api/routes/performances.py` | Add select-frame, reset-tracking endpoints; update stop + status |
| `backend/app/models/performance.py` | Add start_timestamp_ms, click_prompts; add TrackingFrame model |
| `backend/app/schemas/performance.py` | Add SelectFrameRequest, update UploadResponse + PerformanceResponse |
| `backend/app/tasks.py` | Add dispatch_transcode, dispatch_sam2 functions |
| `worker/app/models/performance.py` | Add TrackingFrame model, Performance columns |
| `worker/app/tasks/video_pipeline.py` | Read tracking_frames, use cropped pose, adjust duration for start_ms |
| `worker/app/pipeline/pose.py` | Add run_pose_estimation_cropped(), remove multi + detection |
| `worker/app/pipeline/pose_config.py` | Remove tracker config vars, add SAM2_FRAME_SKIP |
| `worker/app/pipeline/beat_detection.py` | Accept start_ms parameter |
| `worker/Dockerfile` | Remove Re-ID model export step, remove onnx from requirements |
| `frontend/src/App.tsx` | Replace select-dancers route with select-frame |
| `frontend/src/pages/Dashboard.tsx` | Handle transcoding/uploaded/tracking statuses |
| `frontend/src/pages/ProcessingStatus.tsx` | Tracking stage, redirect uploaded→select-frame, stop for tracking |
| `frontend/src/api/performances.ts` | Add selectFrame/resetTracking API, update types |

### Edge Cases & Error Handling

1. **SAM 2 loses a dancer mid-video:** mask_iou drops. tracking_frames stores the confidence. Pose pipeline skips frames where mask_iou < 0.3 or bbox area is zero.

2. **Dancers leave and re-enter frame:** SAM 2 handles natively — mask has zero pixels when off-screen, re-acquires on re-entry.

3. **User clicks on wrong person:** Remove button in UI before submitting. POST `/reset-tracking` to start over after seeing poor results.

4. **SAM 2 memory for long videos:** Base+ peaks ~6-7GB for ~8600 frames on 12GB GPU. Chunked processing (1000 frames with 50 overlap) is the fallback.

5. **Overlapping dancers:** SAM 2's strength — pixel-level masks distinguish overlapping people.

6. **Beat detection start time:** `run_beat_analysis()` receives `start_ms`. Analyzes audio from that point forward.

7. **WHAM 3D compatibility:** Reads poses from frames table. Frames store absolute `timestamp_ms`, so temporal modeling works regardless of start offset.

8. **SAM 2 worker failure:** Exception handler sets status to `failed`. User calls `/reset-tracking` then re-selects.

9. **Cancel during tracking:** `/stop` handles `tracking` status. SAM 2 checks Redis cancel flag every 50 frames. On cancel: stores partial tracking_frames, dispatches pose pipeline on partial data.

10. **HEVC upload:** Transcode runs async. Status `transcoding` until done. Frontend disables video player until `uploaded`.

11. **GPU contention:** All GPU workers at concurrency=1. SAM 2 + main worker never overlap for same performance. Different performances queue sequentially per worker.

12. **Partial tracking + coverage check:** If tracking was cancelled early, coverage is measured against actual tracking_frames extent, not full video duration.

13. **Migration safety:** In-flight performances (status `queued`/`detecting`/`awaiting_selection`) set to `failed` with "Pipeline upgraded" message during migration.

### Testing Strategy

1. **Offline SAM 2 test:** Point SAM 2 at test video, provide 3 click prompts at frame ~150, propagate 500 frames, verify 3 distinct masks with reasonable bboxes.

2. **Coordinate transform test:** Crop known image at known bbox with padding, run RTMPose, transform back, verify keypoints match full-frame RTMPose within 2px.

3. **E2E test with 3-dancer video:** Upload, scrub to ~5s, click 3 standing dancers, verify all tracked through V-formation at ~31s with >95% coverage each.

4. **Frontend click test:** Click on known point, confirm stored normalized coordinate matches actual video position.

5. **Transcode test:** Upload HEVC video, verify transcode completes and video is playable before select-frame allows interaction.

6. **Cancel test:** Start tracking, cancel mid-way, verify partial results available and pose pipeline runs on what exists.

7. **Frame accuracy test:** Compare timestamps of tracking_frames vs actual video frame timestamps to verify seeking accuracy.

### Success Criteria

1. All 3 standing dancers tracked through entire video (>95% frame coverage each)
2. Sitting people never tracked (user doesn't click on them)
3. No skeleton swapping between dancers
4. V-formation transition at ~31s does NOT cause tracking loss
5. Total processing time < 30 minutes for the 289s test video
6. Skeleton overlay correctly positioned on each dancer in review page
7. User can cancel tracking and re-select dancers
8. Single-dancer upload works through click-to-select flow
9. Existing completed performances still viewable after migration

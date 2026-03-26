# Bharatanatyam Dance Analyzer - Handoff Notes

## Project Overview
AI-powered Bharatanatyam dance form analysis tool. Upload practice videos and receive:
- Multi-dancer detection with appearance-based identification (clothing color)
- Occlusion-robust tracking (IoU + centroid + size gating + velocity prediction + appearance + group coherence)
- Real-time pose skeleton overlay (color-coded per dancer, 133 keypoints)
- Per-dancer joint angle analysis (23 body+feet, 42 hands, 68 face)
- AI coaching feedback via Claude API (per-dancer personalized)
- Numeric scoring with clickable breakdowns and improvement tips
- Movement timeline with synchronicity indicators
- Automatic HEVC→H.264 transcode for browser playback (GPU-accelerated)

## Architecture
- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS (port 5173)
- **Backend**: FastAPI + SQLAlchemy 2.0 + Alembic (PostgreSQL 16, port 8000)
- **Worker**: Celery + Redis + RTMPose WholeBody + FFmpeg (NVDEC/NVENC GPU, CUDA 12.4)
- **LLM**: Claude API for coaching synthesis
- **Infra**: Podman-compose with CDI GPU passthrough on Fedora (SELinux)

## Pipeline Flow
1. **Upload** → saved to `/app/uploads/`
2. **Transcode** → `ensure_browser_playable()` detects HEVC/VP9/AV1, GPU-transcodes to H.264+AAC (CUVID decode + NVENC encode, CPU fallback)
3. **Detection** → first 50 frames, extracts all persons with appearance (dominant colors, 72-bin HSV histogram)
4. **Tracking** → 6-layer tracker assigns stable IDs (size gating, velocity prediction, appearance matching, group coherence, extended memory, graveyard re-ID)
5. **Selection** → User selects and names dancers on `/select-dancers/:id` (sees color swatches per dancer)
6. **Pipeline** → Per-dancer pose estimation (tracker seeded with detection bboxes + histograms + group IDs)
7. **Analysis** → Per-frame angles stored, aggregate statistics computed
8. **Coaching** → Claude API generates per-dancer feedback with pose summary context
9. **Scoring** → Numeric 0-100 scores (aramandi, upper body, symmetry, foot, overall)
10. **Review** → Video with skeleton overlay, seekbar, timeline, score breakdowns

### Tracker Occlusion Handling (tracker.py)
The tracker was specifically hardened against walk-through occlusion (someone walking in front of the camera):

| Layer | Parameter | Purpose |
|-------|-----------|---------|
| Size gating | `max_size_ratio=2.5` | Bystander near camera has much larger bbox → rejected |
| Velocity prediction | `velocity_smoothing=0.3` | Predicts position during missing frames along smoothed velocity |
| Appearance matching | `appearance_weight=0.3` | HSV histogram similarity blended with IoU; "red top" ≠ "black top" |
| Group coherence | threshold 0.4 | Tracks pairwise distances between dancers; rejects matches that break formation |
| Extended memory | `max_missing=90` | Tracks survive 3s invisible with position projected along velocity |
| Graveyard re-ID | `graveyard_frames=300` | Expired tracks recoverable for 10s via identity scoring |
| Biometric signatures | 6 body-proportion ratios | Pose-invariant, scale-invariant limb ratios unique per individual |
| Re-ID CNN embeddings | MobileNetV3-Small 512-dim | Visual identity encoding via ONNX on CUDA, baked into Docker image |

### Identity Recovery System (biometrics.py + reid.py)
When tracking is lost, dancers are re-identified by *who they are* rather than *where they were*:

**Biometric Signatures** (`biometrics.py`):
- 6 body-proportion ratios: shoulder/hip width, torso/leg length, upper/lower arm, thigh/shin, head/shoulder, arm/body
- Extracted from existing 133-keypoint pose data (zero additional inference cost)
- EMA-smoothed per track, carried through active → graveyard → re-ID lifecycle
- Discriminates different body types even in matching costumes

**Re-ID CNN Embeddings** (`reid.py`):
- MobileNetV3-Small backbone producing 512-dim L2-normalized visual embeddings
- ONNX model baked into Docker image at build time via `export_reid_model.py` (~5MB)
- Runs on existing onnxruntime-gpu CUDA EP, ~2ms per person crop
- Captures texture, silhouette, and visual features beyond color histograms
- Singleton pattern, lazy-loaded on first use

**Identity Scoring** (in `tracker.py`):
- `_identity_score()` blends: histogram (0.25) + biometric (0.25) + Re-ID (0.30)
- Weights adapt when signals are unavailable (graceful degradation)
- Graveyard re-ID: 20% proximity + 80% identity (was 50/50 proximity/appearance)
- Re-lock: full identity scoring with 3-frame confirmation
- `reseed()`: identity-based greedy matching instead of reverting to frame-1 positions

**Config**: `REID_ENABLED=true/false` env var to toggle Re-ID CNN (biometrics always on)

### Appearance System (appearance.py)
- Extracts from upper 60% of bbox (torso region), trimmed 10% margins
- 72-bin HSV histogram for tracker matching (exponential moving average)
- Dominant colors with human-readable names for UI ("red and black top")
- Stored on `DetectedPerson.appearance` (JSON) and `DetectedPerson.color_histogram` (JSON)
- During full pipeline, appearance extracted every 5th frame

### Early Stop
- User can stop processing at any point during pose estimation
- `POST /performances/{id}/stop` sets Redis flag `cancel:{id}`
- Worker checks every 10 frames, stops reading video, continues through full analysis/scoring on collected frames

## Pose Analysis Metrics
**From body+feet (23 keypoints):**
- Knee angles (aramandi), torso uprightness, arm extension, hip symmetry
- Foot turnout, foot flatness, foot flexion angle
- Shoulder elevation (ear-to-shoulder, normalized by torso)
- Neck lateral tilt (attami — shoulder midpoint to nose)

**From face (68 landmarks):**
- Head lateral tilt (shirobheda — chin to nose bridge angle)
- Head forward tilt (nose tip to chin ratio)

**From hands (42 keypoints):**
- Wrist flexion (elbow → wrist → middle finger base)
- Finger extension (PIP angles for 4 fingers + thumb per hand)

**From WHAM 3D (24 SMPL joints, optional — when installed):**
- 3D knee angle (camera-invariant aramandi), 3D torso angle (gravity-relative)
- 3D arm extension, hip abduction, hip symmetry (catches rotational asymmetry)
- Torso twist (shoulder vs hip plane in XZ)
- Foot-ground contact probability (left/right heel + toe)

All metrics stored in `JointAngleState.all_angles` JSON per frame.

## Audio / Beat Detection
- `worker/app/pipeline/beat_detection.py` — librosa onset detection + foot-strike correlation
- Extracts audio via FFmpeg → detects percussive onsets (spectral flux) → estimates tempo BPM
- Foot strikes detected from negative derivative peaks on smoothed foot_flatness time series
- Rhythm sync score: 60% match rate + 40% timing precision (75ms tolerance), scaled 0-100
- Beat data stored on Performance: `beat_timestamps` (JSON), `tempo_bpm` (float)
- Frontend: beat markers bar in DanceTimeline, rhythm score card in ScoreCards

## WHAM 3D Integration (Separate Container)
- **Architecture**: Separate `wham_worker` container (PyTorch 1.13.1 + CUDA 11.6) communicates via Celery
- **Main worker** (`worker/app/pipeline/wham.py`): thin dispatch client, sends fire-and-forget tasks to `wham_3d` queue
- **WHAM worker** (`wham_worker/`): receives tasks, reads 2D poses from Postgres, runs WHAM 3D, writes `joints_3d`/`world_position`/`foot_contact` back to frame rows
- **Staged GPU loading**: HMR2a feature extractor (~2.9GB) → extract → release → WHAM network (~0.2GB) → infer → release (fits 12GB VRAM)
- **Bypasses ViTPose**: Feeds RTMPose 2D keypoints directly via `_poses_to_tracking_results()`, uses HMR2 only for image features
- **Fire-and-forget**: Main pipeline completes with 2D scores immediately; WHAM enriches frames asynchronously
- Scoring prefers 3D values with `or` fallback to 2D
- `technique_scores.inputs.source_3d` flag indicates 3D data was used
- **Model files**: `$HOME/bharatanatyam-data/smpl/checkpoints/` (wham_vit_bedlam_w_3dpw.pth.tar, hmr2a.ckpt) and `$HOME/bharatanatyam-data/smpl/dataset/body_models/` (SMPL_NEUTRAL.pkl)

## Score System (all 0-100)
| Score | Weight | What It Measures |
|-------|--------|------------------|
| Aramandi | 30% | Knee angle (ideal 105°) + consistency |
| Upper Body | 20% | Torso deviation from vertical (ideal 0°) |
| Symmetry | 25% | Hip symmetry + arm/foot balance |
| Foot Technique | 25% | Turnout angle (ideal 45-60°) + flatness |
| Overall | — | Weighted composite |

Score cards are clickable → show raw measurements, ideal values, actionable tips.

## Database Tables (10)
`dancers`, `sessions`, `performances`, `frames`, `joint_angle_states`, `balance_metrics`, `mudra_states`, `analyses`, `detected_persons`, `performance_dancers`

All FKs have ON DELETE CASCADE at both ORM and DB levels.

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/performances/` | List performances (optional `dancer_id` filter) |
| GET | `/api/performances/{id}` | Performance metadata + analysis (no frames) |
| GET | `/api/performances/{id}/frames` | All frames with pose data |
| GET | `/api/performances/{id}/timeline` | Per-frame angles/balance for timeline |
| GET | `/api/performances/{id}/status` | Polling during processing |
| POST | `/api/performances/{id}/stop` | Early stop processing |
| DELETE | `/api/performances/{id}` | Cascade delete |
| GET | `/api/performances/{id}/detected-persons` | Detection results (includes appearance) |
| POST | `/api/performances/{id}/select-dancers` | Dancer selection |
| POST | `/api/upload/` | Video upload |
| GET/POST | `/api/dancers/` | Dancer CRUD |

## First-Time Setup

```bash
# 1. Copy environment config
cp .env.example .env
# Edit .env to set ANTHROPIC_API_KEY

# 2. Configure GPU passthrough (if Nvidia)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 3. Build and start all services
podman-compose build
podman-compose up -d

# 4. Run database migrations
podman exec -e PYTHONPATH=/app -w /app source1_api_1 alembic upgrade head
```

## Development Notes
- Worker doesn't auto-reload; needs `podman restart source1_worker_1` after code changes
- API container has `--reload` (uvicorn watches files)
- Frontend has Vite HMR via volume mount
- SELinux: `~/.config/containers/containers.conf` with `[containers] label = false`
- Data volumes: `$HOME/bharatanatyam-data/postgres` and `redis`

## File Structure
```
bharatanatyam-analyzer/
├── docker-compose.yml
├── .env.example
├── HANDOFF.md
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Settings
│   │   ├── database.py          # SQLAlchemy engine
│   │   ├── tasks.py             # Celery dispatcher
│   │   ├── models/
│   │   │   ├── dancer.py
│   │   │   ├── performance.py   # Performance, DetectedPerson, PerformanceDancer
│   │   │   └── analysis.py      # Frame, JointAngleState, BalanceMetrics, MudraState, Analysis
│   │   ├── schemas/performance.py
│   │   └── api/routes/
│   │       ├── dancers.py
│   │       ├── performances.py  # List, get, frames, timeline, status, stop, delete, select
│   │       └── upload.py
│   ├── alembic/
│   └── requirements.txt
├── worker/
│   ├── app/
│   │   ├── celery_app.py
│   │   ├── db.py
│   │   ├── tasks/
│   │   │   ├── video_pipeline.py   # Main pipeline (single + multi-dancer)
│   │   │   └── detect_dancers.py   # Detection pass (first 50 frames)
│   │   ├── pipeline/
│   │   │   ├── ingest.py           # FFprobe metadata + HEVC→H.264 transcode
│   │   │   ├── pose.py             # RTMPose WholeBody 133-pt + NVDEC
│   │   │   ├── angles.py           # Joint angle computation (2D + 3D)
│   │   │   ├── scoring.py          # Numeric scoring (0-100, prefers 3D)
│   │   │   ├── tracker.py          # 8-layer occlusion-robust tracker with identity recovery
│   │   │   ├── appearance.py       # Color histogram + dominant color extraction
│   │   │   ├── biometrics.py       # Body-proportion signature extraction + similarity
│   │   │   ├── reid.py             # Re-ID CNN embeddings (MobileNetV3-Small ONNX)
│   │   │   ├── llm.py              # Claude API coaching
│   │   │   ├── beat_detection.py   # Audio onset detection + rhythm scoring
│   │   │   └── wham.py             # WHAM 3D dispatch client (fire-and-forget)
│   │   └── models/
│   └── requirements.txt
├── wham_worker/                     # Separate WHAM 3D container (PyTorch 1.13.1)
│   ├── Dockerfile
│   └── app/
│       ├── celery_app.py            # Celery on wham_3d queue
│       ├── db.py                    # Postgres session
│       ├── models.py                # Minimal Frame/Performance models
│       ├── inference.py             # WHAM inference (staged GPU loading)
│       └── tasks.py                 # run_wham_3d task
└── frontend/
    └── src/
        ├── App.tsx
        ├── pages/
        │   ├── Dashboard.tsx          # Upload + performances list
        │   ├── ProcessingStatus.tsx    # Pipeline progress + stop button
        │   ├── DancerSelection.tsx     # Multi-dancer selection with color swatches
        │   └── VideoReview.tsx         # Skeleton overlay, seekbar, timeline, scores, coaching
        ├── components/Layout.tsx
        └── api/
            ├── client.ts              # Axios instance
            ├── dancers.ts
            └── performances.ts        # All performance API calls + types
```

## Roadmap
| Feature | Status |
|---------|--------|
| Skeleton overlay + coaching | Done |
| Multi-dancer detection/tracking | Done |
| Per-dancer scoring with breakdowns | Done |
| Stop & partial analysis | Done |
| Video scrubber + skeleton toggles | Done |
| Movement timeline + synchronicity | Done |
| Extended pose metrics (head/wrist/fingers/shoulders/neck) | Done |
| Audio/beat detection + rhythm scoring | Done |
| WHAM 3D integration (data model + angles + scoring) | Done |
| WHAM 3D worker container (separate PyTorch 1.13.1 container) | Done |
| Occlusion-robust tracking (appearance + velocity + group coherence) | Done |
| Identity recovery (biometric signatures + Re-ID CNN embeddings) | Done |
| Video transcode (HEVC→H.264, GPU-accelerated) | Done |
| WHAM 3D end-to-end test | Not yet tested |
| WHAM foot contact rhythm (Phase 2b) | Not started |
| WHAM global trajectory / stage coverage (Phase 3) | Not started |
| Mudra classification | Not started (table + finger data ready) |
| Adavu classification | Not started |
| Multi-camera 3D reconstruction | Not started |

## Recent Changes (2026-03-25)
- **Identity recovery system**: Dual-layer biometric body-proportion signatures (6 ratios from pose keypoints) + MobileNetV3-Small Re-ID CNN embeddings (512-dim via ONNX on CUDA). Graveyard re-ID now scores 20% proximity + 80% identity. `reseed()` uses identity-based greedy matching instead of reverting to frame-1 positions. Re-lock uses full identity scoring. Re-ID model baked into Docker image at build time.
- **New files**: `biometrics.py`, `reid.py`, `export_reid_model.py`
- **Config**: `REID_ENABLED` and `REID_MODEL_PATH` env vars in `pose_config.py`

## Changes (2026-03-23)
- **Occlusion-robust tracker**: Replaced simple IoU+centroid tracker with 6-layer system (size gating, velocity prediction, appearance matching, group coherence, extended memory, graveyard re-ID)
- **Appearance extraction**: New `appearance.py` — extracts dominant colors and HSV histograms from torso region per detected person
- **Dancer selection UI**: Now shows color swatches and descriptions ("red and black top") per detected person
- **Video transcode**: Auto-transcodes HEVC/VP9/AV1 → H.264+AAC at start of detection task (GPU NVENC with CPU fallback)
- **DB migration**: `e1a2b3c4d5f6` added `appearance` and `color_histogram` columns to `detected_persons`
- **Bugfix**: `fmt()` in VideoReview.tsx crashed on `null` values (`!== undefined` → `!= null`)

## Known Issues / Watch Items
- Tracker re-ID score threshold is 0.3 — may need tuning if false re-IDs occur
- Group coherence threshold is 0.4 — dancers who move far apart (stage entrances) may trigger false rejection
- Frame count per dancer may be low if occlusion is extreme — tracker discards frames it can't confidently assign
- Re-ID model is MobileNetV3-Small pretrained on ImageNet (not person re-ID). Works for visual feature encoding but could be improved with fine-tuning on person re-ID data (e.g., Market-1501, MSMT17)
- Biometric ratios are 2D-projected — deep aramandi changes apparent limb ratios. EMA smoothing mitigates this; WHAM 3D joints could provide true 3D biometrics in future
- `REID_ENABLED=false` disables CNN embeddings (biometric ratios still active)

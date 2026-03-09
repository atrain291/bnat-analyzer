# Bharatanatyam Dance Analyzer - Handoff Notes

## Project Overview
AI-powered Bharatanatyam dance form analysis tool. Upload practice videos and receive:
- Multi-dancer detection and tracking (IoU + centroid + positional fallback)
- Real-time pose skeleton overlay (color-coded per dancer, 133 keypoints)
- Per-dancer joint angle analysis (23 body+feet, 42 hands, 68 face)
- AI coaching feedback via Claude API (per-dancer personalized)
- Numeric scoring with clickable breakdowns and improvement tips
- Movement timeline with synchronicity indicators

## Architecture
- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS (port 5173)
- **Backend**: FastAPI + SQLAlchemy 2.0 + Alembic (PostgreSQL 16, port 8000)
- **Worker**: Celery + Redis + RTMPose WholeBody + FFmpeg (NVDEC GPU, CUDA 12.4)
- **LLM**: Claude API for coaching synthesis
- **Infra**: Podman-compose with CDI GPU passthrough on Fedora (SELinux)

## Pipeline Flow
1. **Upload** → `dispatch_detection` runs first 50 frames detecting all persons
2. **Detection** → IoU+centroid tracker with positional fallback assigns stable track IDs
3. **Selection** → User selects and names dancers on `/select-dancers/:id`
4. **Pipeline** → Per-dancer pose estimation (tracker seeded with detection bboxes)
5. **Analysis** → Per-frame angles stored, aggregate statistics computed
6. **Coaching** → Claude API generates per-dancer feedback with pose summary context
7. **Scoring** → Numeric 0-100 scores (aramandi, upper body, symmetry, foot, overall)
8. **Review** → Video with skeleton overlay, seekbar, timeline, score breakdowns

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

All metrics stored in `JointAngleState.all_angles` JSON per frame.

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
| GET | `/api/performances/{id}/detected-persons` | Detection results |
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
│   │   │   ├── ingest.py           # FFprobe metadata
│   │   │   ├── pose.py             # RTMPose WholeBody 133-pt + NVDEC
│   │   │   ├── angles.py           # Joint angle computation (body+face+hands)
│   │   │   ├── scoring.py          # Numeric scoring (0-100)
│   │   │   ├── tracker.py          # IoU + centroid + positional tracker
│   │   │   └── llm.py              # Claude API coaching
│   │   └── models/
│   └── requirements.txt
└── frontend/
    └── src/
        ├── App.tsx
        ├── pages/
        │   ├── Dashboard.tsx          # Upload + performances list
        │   ├── ProcessingStatus.tsx    # Pipeline progress + stop button
        │   ├── DancerSelection.tsx     # Multi-dancer selection
        │   └── VideoReview.tsx         # Skeleton overlay, seekbar, timeline, scores, coaching
        ├── components/Layout.tsx
        └── api/
            ├── client.ts              # Axios instance
            ├── dancers.ts
            └── performances.ts        # All performance API calls
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
| Mudra classification | Not started (table + finger data ready) |
| Adavu classification | Not started |
| Music synchronicity | Not started (needs audio/beat detection) |
| Multi-camera 3D reconstruction | Not started |

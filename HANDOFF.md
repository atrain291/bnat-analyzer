# Bharatanatyam Dance Analyzer - Handoff Notes

## Project Overview
AI-powered Bharatanatyam dance form analysis tool. Upload practice videos and receive:
- Real-time pose skeleton overlay (saffron/gold color scheme)
- Joint angle analysis (aramandi depth, torso uprightness, arm extension)
- AI coaching feedback via Claude API
- Balance and symmetry metrics

## Architecture
Same architecture as fencing-analyzer, adapted for Bharatanatyam:
- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS
- **Backend**: FastAPI + SQLAlchemy + Alembic (PostgreSQL)
- **Worker**: Celery + Redis + YOLOv8-Pose + FFmpeg (NVDEC GPU)
- **LLM**: Claude API for coaching synthesis
- **Infra**: Podman containers with CDI GPU passthrough

## Key Domain Adaptations from Fencing Analyzer
1. **Fencer -> Dancer**: Renamed entity with experience_level field
2. **Bout -> Performance**: Dance-specific metadata (item_name, item_type, talam, ragam)
3. **Opponent tracking removed**: Single dancer focus
4. **Bharatanatyam-specific analysis tables**:
   - JointAngleState: aramandi_angle, torso_uprightness, arm_extension, hip_symmetry
   - BalanceMetrics: center_of_mass, weight_distribution, stability_score
   - MudraState: hand gesture classification (Stage 2+)
5. **Scoring dimensions**: aramandi, upper_body, symmetry, rhythm_consistency
6. **UI theme**: Saffron/gold (#F9A825) instead of cyan, lotus flower icon
7. **LLM prompt**: Bharatanatyam guru persona with dance-specific coaching

## Development Stages (Roadmap)

| Stage | Status | Features |
|-------|--------|----------|
| 1 | Complete (code) | Skeleton overlay, Claude API coaching feedback |
| 2 | Planned | Mudra (hand gesture) classification using custom model |
| 3 | Planned | Joint angle computation, aramandi depth scoring |
| 4 | Planned | Adavu classification and talam synchronization |
| 5 | Planned | Reference pose comparison, technique scoring |
| 6 | Planned | Abhinaya (facial expression) analysis |
| 7 | Planned | Longitudinal tracking across sessions |
| 8 | Planned | Live camera feed |

## First-Time Setup

```bash
# 1. Copy environment config
cp .env.example .env
# Edit .env to set ANTHROPIC_API_KEY

# 2. Configure GPU passthrough (if Nvidia)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 3. Build containers
podman build --security-opt seccomp=unconfined --security-opt label=disable -t bharatanatyam-frontend frontend/
podman build --security-opt seccomp=unconfined --security-opt label=disable -t bharatanatyam-backend backend/
podman build --security-opt seccomp=unconfined --security-opt label=disable -t bharatanatyam-worker worker/

# 4. Start all services
podman-compose up -d

# 5. Run database migrations
podman exec bharatanatyam_api alembic revision --autogenerate -m "initial"
podman exec bharatanatyam_api alembic upgrade head
```

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
│   │   ├── models/              # ORM models
│   │   │   ├── dancer.py
│   │   │   ├── performance.py
│   │   │   └── analysis.py
│   │   ├── schemas/             # Pydantic schemas
│   │   │   ├── dancer.py
│   │   │   └── performance.py
│   │   └── api/routes/
│   │       ├── dancers.py
│   │       ├── performances.py
│   │       └── upload.py
│   ├── alembic/
│   └── requirements.txt
├── worker/
│   ├── app/
│   │   ├── celery_app.py
│   │   ├── db.py
│   │   ├── tasks/video_pipeline.py
│   │   ├── pipeline/
│   │   │   ├── ingest.py        # FFprobe metadata
│   │   │   ├── pose.py          # YOLOv8-Pose + NVDEC
│   │   │   └── llm.py           # Claude API coaching
│   │   └── models/
│   └── requirements.txt
└── frontend/
    └── src/
        ├── App.tsx
        ├── pages/
        │   ├── Dashboard.tsx       # Upload UI with dance metadata
        │   ├── ProcessingStatus.tsx # Pipeline progress
        │   └── VideoReview.tsx     # Skeleton overlay + coaching
        ├── components/Layout.tsx
        └── api/                    # Axios API client
```

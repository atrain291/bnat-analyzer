# SAM 2 Tracking Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken bbox-based tracker with SAM 2 pixel-level video segmentation, enabling reliable multi-dancer tracking through a click-to-select interface.

**Architecture:** SAM 2 runs in a separate container (`sam2_worker`), receives click prompts from the user, segments dancers through the video, and stores per-frame bboxes. The main worker then runs RTMPose on pre-cropped dancer images. No tracker code needed — SAM 2 handles identity persistence at the pixel level.

**Tech Stack:** SAM 2.1 (sam2 PyPI package), PyTorch 2.5.1 + CUDA 12.4, Celery + Redis, SQLAlchemy 2.0, PostgreSQL 16, React 18 + TypeScript + Vite

**Spec:** `docs/superpowers/specs/2026-03-27-sam2-tracking-integration-design.md`

---

## Phase 1: Database & Backend Foundation

### Task 1: Database Migration — Add tracking_frames table and Performance columns

**Files:**
- Create: `backend/alembic/versions/f1a2b3c4d5e6_add_sam2_tracking.py`
- Modify: `backend/app/models/performance.py`
- Modify: `worker/app/models/performance.py`

- [ ] **Step 1: Add columns and table to backend ORM model**

In `backend/app/models/performance.py`, add to the `Performance` class:
```python
start_timestamp_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
click_prompts: Mapped[list | None] = mapped_column(JSON, nullable=True)
```

Add new `TrackingFrame` class after `DetectedPerson`:
```python
class TrackingFrame(Base):
    __tablename__ = "tracking_frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id", ondelete="CASCADE"), index=True)
    dancer_index: Mapped[int] = mapped_column(Integer)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[dict] = mapped_column(JSON)
    mask_iou: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        Index("ix_tracking_frames_perf_dancer_ts", "performance_id", "dancer_index", "timestamp_ms"),
    )
```

Import `Index` from sqlalchemy at the top of the file.

- [ ] **Step 2: Mirror changes in worker ORM model**

In `worker/app/models/performance.py`, add the same `TrackingFrame` class and Performance columns. Follow the exact same pattern.

- [ ] **Step 3: Generate Alembic migration**

```bash
podman exec source1_api_1 bash -c "cd /app && PYTHONPATH=/app alembic revision --autogenerate -m 'Add SAM 2 tracking tables'"
```

- [ ] **Step 4: Edit the migration to add data migration for in-flight performances**

In the generated migration file, add after the table creation:
```python
from alembic import op
import sqlalchemy as sa

def upgrade():
    # ... auto-generated table/column creation ...

    # Migrate in-flight performances to failed
    op.execute(
        "UPDATE performances SET status = 'failed', "
        "error = 'Pipeline upgraded to SAM 2 tracking — please re-upload this video' "
        "WHERE status IN ('queued', 'detecting', 'awaiting_selection')"
    )
```

- [ ] **Step 5: Run migration**

```bash
podman-compose up -d postgres
sleep 3
podman exec -e PYTHONPATH=/app -w /app source1_api_1 alembic upgrade head
```

- [ ] **Step 6: Verify**

```bash
podman exec source1_postgres_1 psql -U bharatanatyam -d bharatanatyam_analyzer -c "\d tracking_frames"
podman exec source1_postgres_1 psql -U bharatanatyam -d bharatanatyam_analyzer -c "\d performances" | grep -E "start_timestamp|click_prompts"
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/performance.py worker/app/models/performance.py backend/alembic/versions/
git commit -m "Add tracking_frames table and Performance columns for SAM 2"
```

### Task 2: Backend API — Transcode task dispatch and select-frame endpoint

**Files:**
- Create: `worker/app/tasks/transcode.py`
- Modify: `backend/app/api/routes/upload.py`
- Modify: `backend/app/api/routes/performances.py`
- Modify: `backend/app/schemas/performance.py`
- Modify: `backend/app/tasks.py`

- [ ] **Step 1: Create the transcode Celery task**

Create `worker/app/tasks/transcode.py`:
```python
import logging

from app.celery_app import app
from app.db import get_session
from app.models.performance import Performance
from app.pipeline.ingest import ensure_browser_playable

logger = logging.getLogger(__name__)


@app.task(bind=True, name="worker.app.tasks.transcode.run_transcode")
def run_transcode(self, performance_id: int, video_path: str):
    logger.info(f"Transcoding performance {performance_id}: {video_path}")
    try:
        ensure_browser_playable(video_path)
        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "uploaded"
        logger.info(f"Transcode complete for performance {performance_id}")
    except Exception as e:
        logger.error(f"Transcode failed for performance {performance_id}: {e}")
        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "failed"
                perf.error = f"Video transcode failed: {str(e)[:500]}"
```

- [ ] **Step 2: Add dispatch functions to backend/app/tasks.py**

Add `dispatch_transcode` and `dispatch_sam2` functions:
```python
def dispatch_transcode(performance_id: int, video_path: str) -> str:
    result = celery_app.send_task(
        "worker.app.tasks.transcode.run_transcode",
        args=[performance_id, video_path],
        queue="video_pipeline",
    )
    return result.id


def dispatch_sam2(performance_id: int, video_path: str, start_timestamp_ms: int,
                  click_prompts: list, selected_tracks: list) -> str:
    result = celery_app.send_task(
        "sam2_worker.app.tasks.run_sam2_tracking",
        args=[performance_id, video_path, start_timestamp_ms, click_prompts, selected_tracks],
        queue="sam2_tracking",
    )
    return result.id
```

- [ ] **Step 3: Add Pydantic schemas**

In `backend/app/schemas/performance.py`, add:
```python
class ClickPrompt(BaseModel):
    x: float
    y: float
    label: str | None = None

class SelectFrameRequest(BaseModel):
    start_timestamp_ms: int
    prompts: list[ClickPrompt]
```

Add `has_3d` and `start_timestamp_ms` to `PerformanceResponse` (if not already present).

- [ ] **Step 4: Modify upload endpoint**

In `backend/app/api/routes/upload.py`, change:
- Import `dispatch_transcode` instead of `dispatch_detection`
- Set `status="transcoding"` instead of `status="queued"`
- Call `dispatch_transcode(performance.id, video_path)` instead of `dispatch_detection`

- [ ] **Step 5: Add select-frame endpoint**

In `backend/app/api/routes/performances.py`, add:
```python
@router.post("/{performance_id}/select-frame")
def select_frame(performance_id: int, body: SelectFrameRequest, db: Session = Depends(get_db)):
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    if performance.status != "uploaded":
        raise HTTPException(status_code=400, detail=f"Performance not ready (status: {performance.status})")
    if not body.prompts:
        raise HTTPException(status_code=400, detail="At least one dancer must be selected")

    # Store click prompts
    performance.start_timestamp_ms = body.start_timestamp_ms
    performance.click_prompts = [p.model_dump() for p in body.prompts]

    # Create PerformanceDancer records
    selected_tracks = []
    for i, prompt in enumerate(body.prompts):
        pd = PerformanceDancer(
            performance_id=performance_id,
            track_id=i,
            label=prompt.label,
        )
        db.add(pd)
        db.flush()
        selected_tracks.append({
            "track_id": i,
            "performance_dancer_id": pd.id,
            "label": prompt.label,
        })

    # Commit DB changes FIRST, then dispatch (prevents race if dispatch fails)
    performance.status = "tracking"
    db.commit()

    # Dispatch SAM 2 tracking
    video_path = f"/app/uploads/{performance.video_key}"
    from app.tasks import dispatch_sam2
    dispatch_sam2(performance_id, video_path, body.start_timestamp_ms,
                  [p.model_dump() for p in body.prompts], selected_tracks)

    return {"status": "tracking", "dancers_selected": len(body.prompts)}
```

- [ ] **Step 6: Add reset-tracking endpoint**

```python
@router.post("/{performance_id}/reset-tracking")
def reset_tracking(performance_id: int, db: Session = Depends(get_db)):
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    if performance.status not in ("uploaded", "tracking", "failed"):
        raise HTTPException(status_code=400, detail=f"Cannot reset (status: {performance.status})")

    # Cancel if tracking, then delete ALL related data (frames, analyses, tracking)
    if performance.status == "tracking":
        import redis as r
        redis_client = r.from_url(os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"))
        redis_client.set(f"cancel:{performance_id}", "1", ex=300)

    # Delete ALL related data: tracking, frames, analyses, balance, angles, dancers
    from app.models.performance import TrackingFrame
    from app.models.analysis import Frame, Analysis, JointAngleState, BalanceMetrics
    # Delete in FK-safe order (children first)
    frame_ids = [f.id for f in db.query(Frame.id).filter(Frame.performance_id == performance_id).all()]
    if frame_ids:
        db.query(JointAngleState).filter(JointAngleState.frame_id.in_(frame_ids)).delete(synchronize_session=False)
        db.query(BalanceMetrics).filter(BalanceMetrics.frame_id.in_(frame_ids)).delete(synchronize_session=False)
    db.query(Frame).filter(Frame.performance_id == performance_id).delete()
    db.query(Analysis).filter(Analysis.performance_id == performance_id).delete()
    db.query(TrackingFrame).filter(TrackingFrame.performance_id == performance_id).delete()
    db.query(PerformanceDancer).filter(PerformanceDancer.performance_id == performance_id).delete()

    performance.start_timestamp_ms = None
    performance.click_prompts = None
    performance.status = "uploaded"
    performance.error = None
    db.commit()

    return {"status": "uploaded"}
```

- [ ] **Step 7: Update stop endpoint to handle tracking status**

In the existing `stop_performance` function, add `"tracking"` to the list of stoppable statuses.

- [ ] **Step 8: Update status endpoint to handle new statuses**

Ensure the status endpoint returns progress for `tracking` and `transcoding` stages.

- [ ] **Step 9: Verify API starts without errors**

```bash
podman-compose up -d api
sleep 5
curl -s http://localhost:8000/docs | head -5
```

- [ ] **Step 10: Commit**

```bash
git add worker/app/tasks/transcode.py backend/app/api/routes/ backend/app/schemas/ backend/app/tasks.py
git commit -m "Add select-frame API, transcode task, and reset-tracking endpoint"
```

---

## Phase 2: SAM 2 Worker Container

### Task 3: SAM 2 worker container setup

**Files:**
- Create: `sam2_worker/Dockerfile`
- Create: `sam2_worker/requirements.txt`
- Create: `sam2_worker/app/__init__.py`
- Create: `sam2_worker/app/celery_app.py`
- Create: `sam2_worker/app/db.py`
- Create: `sam2_worker/app/models.py`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Create sam2_worker directory structure**

```bash
mkdir -p sam2_worker/app
touch sam2_worker/app/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

Create `sam2_worker/requirements.txt`:
```
sam2>=1.0
celery[redis]==5.4.0
redis==5.2.1
sqlalchemy==2.0.36
psycopg2-binary==2.9.10
opencv-python-headless==4.10.0.84
numpy==1.26.4
```

- [ ] **Step 3: Create Dockerfile**

Create `sam2_worker/Dockerfile`:
```dockerfile
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "-Q", "sam2_tracking", "-c", "1", "--prefetch-multiplier=1"]
```

- [ ] **Step 4: Create celery_app.py**

Create `sam2_worker/app/celery_app.py` (follow wham_worker pattern):
```python
import os
from celery import Celery

REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")

app = Celery(
    "sam2_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)
```

- [ ] **Step 5: Create db.py**

Create `sam2_worker/app/db.py` (identical pattern to wham_worker):
```python
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://bharatanatyam:bharatanatyam@postgres:5432/bharatanatyam_analyzer",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=2)
SessionLocal = sessionmaker(bind=engine)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

- [ ] **Step 6: Create models.py**

Create `sam2_worker/app/models.py`:
```python
"""Minimal SQLAlchemy models — only columns the SAM 2 worker reads/writes."""

from sqlalchemy import String, Integer, Float, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Performance(Base):
    __tablename__ = "performances"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(20))
    video_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    start_timestamp_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    click_prompts: Mapped[list | None] = mapped_column(JSON, nullable=True)
    pipeline_progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)


class PerformanceDancer(Base):
    __tablename__ = "performance_dancers"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id"))
    track_id: Mapped[int] = mapped_column(Integer)
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)


class TrackingFrame(Base):
    __tablename__ = "tracking_frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id", ondelete="CASCADE"))
    dancer_index: Mapped[int] = mapped_column(Integer)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[dict] = mapped_column(JSON)
    mask_iou: Mapped[float] = mapped_column(Float)
```

- [ ] **Step 7: Add sam2_worker service to docker-compose.yml**

Add after the `wham_worker` service:
```yaml
sam2_worker:
  build: ./sam2_worker
  volumes:
    - ./sam2_worker/app:/app/app:z
    - uploads:/app/uploads
    - ${HOME}/bharatanatyam-data/sam2:/app/models:z
  env_file:
    - .env
  depends_on:
    - postgres
    - redis
  devices:
    - nvidia.com/gpu=all
  networks:
    - bharatanatyam
  security_opt:
    - label=disable
```

- [ ] **Step 8: Build and verify container starts**

```bash
podman-compose build sam2_worker
podman-compose up -d sam2_worker
sleep 5
podman logs source1_sam2_worker_1
```

Expected: Celery worker starts and connects to Redis. SAM 2 model not loaded yet (loaded on first task).

- [ ] **Step 9: Commit**

```bash
git add sam2_worker/ docker-compose.yml
git commit -m "Add SAM 2 worker container scaffold"
```

### Task 4: SAM 2 inference and tracking task

**Files:**
- Create: `sam2_worker/app/inference.py`
- Create: `sam2_worker/app/tasks.py`

- [ ] **Step 1: Create inference.py — SAM 2 video prediction**

Create `sam2_worker/app/inference.py`:
```python
"""SAM 2.1 video segmentation inference."""

import logging
import subprocess
import json

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT = "/app/models/sam2.1_hiera_base_plus.pt"
MIN_MASK_IOU = 0.1
STORE_EVERY_N = 2  # Store tracking results every N frames (matches POSE_FRAME_SKIP)


def _get_video_info(video_path: str) -> dict:
    """Get video fps, width, height via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    fps_parts = stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    return {
        "fps": fps,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "total_frames": int(stream.get("nb_frames", 0)),
    }


def run_sam2_tracking(
    video_path: str,
    start_timestamp_ms: int,
    click_prompts: list[dict],
    is_cancelled=None,
    progress_callback=None,
) -> list[dict]:
    """Run SAM 2 video segmentation from click prompts.

    Args:
        video_path: Path to video file.
        start_timestamp_ms: Timestamp to start tracking from.
        click_prompts: List of {x, y, label} dicts (normalized 0-1 coords).
        is_cancelled: Callable returning True if task should stop.
        progress_callback: Callable(current_frame, total_frames).

    Returns:
        List of {dancer_index, timestamp_ms, bbox, mask_iou} dicts.
    """
    from sam2.build_sam import build_sam2_video_predictor

    video_info = _get_video_info(video_path)
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    total_frames = video_info["total_frames"]

    start_frame_idx = int(start_timestamp_ms / 1000.0 * fps)
    effective_total = total_frames - start_frame_idx

    logger.info(f"SAM 2 tracking: {len(click_prompts)} dancers, "
                f"start_frame={start_frame_idx}, total_frames={total_frames}, "
                f"video={width}x{height}@{fps:.1f}fps")

    # Load model and init state
    predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device="cuda")
    state = predictor.init_state(video_path=video_path)

    # Add click prompts at the start frame
    for i, prompt in enumerate(click_prompts):
        points = np.array([[prompt["x"] * width, prompt["y"] * height]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        _, _, _ = predictor.add_new_points_or_box(
            state, frame_idx=start_frame_idx, obj_id=i,
            points=points, labels=labels,
        )
        logger.info(f"Added prompt {i} at pixel ({points[0][0]:.0f}, {points[0][1]:.0f})")

    # Propagate forward
    # NOTE: propagate_in_video yields (frame_idx, obj_ids, masks_dict)
    # where masks_dict is {obj_id: mask_tensor}, NOT a list
    results = []
    frame_count = 0
    for frame_idx, obj_ids, masks_dict in predictor.propagate_in_video(state, start_frame_idx=start_frame_idx):
        # Only store every STORE_EVERY_N frames
        if (frame_idx - start_frame_idx) % STORE_EVERY_N != 0:
            continue

        timestamp_ms = int(frame_idx / fps * 1000)

        for obj_id in obj_ids:
            mask = masks_dict[obj_id]
            # mask shape: (1, H, W) — squeeze to (H, W)
            mask_np = mask.squeeze().cpu().numpy() > 0.5

            # Derive bbox from mask
            ys, xs = np.where(mask_np)
            if len(ys) == 0:
                # Dancer not visible — store zero bbox
                results.append({
                    "dancer_index": int(obj_id),
                    "timestamp_ms": timestamp_ms,
                    "bbox": {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0},
                    "mask_iou": 0.0,
                })
                continue

            x_min = float(xs.min()) / width
            y_min = float(ys.min()) / height
            x_max = float(xs.max()) / width
            y_max = float(ys.max()) / height
            # mask_iou: fraction of bbox that is mask (crude confidence proxy)
            bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
            mask_iou = float(len(ys)) / max(bbox_area, 1)

            results.append({
                "dancer_index": int(obj_id),
                "timestamp_ms": timestamp_ms,
                "bbox": {"x_min": round(x_min, 5), "y_min": round(y_min, 5),
                         "x_max": round(x_max, 5), "y_max": round(y_max, 5)},
                "mask_iou": round(mask_iou, 4),
            })

        frame_count += 1
        if progress_callback and frame_count % 50 == 0:
            progress_callback(frame_idx - start_frame_idx, effective_total)

        if is_cancelled and frame_count % 50 == 0 and is_cancelled():
            logger.info(f"SAM 2 tracking cancelled at frame {frame_idx}")
            break

    # Release GPU memory
    del predictor, state
    torch.cuda.empty_cache()

    logger.info(f"SAM 2 tracking complete: {frame_count} frames, {len(results)} tracking entries")
    return results
```

- [ ] **Step 2: Create tasks.py — Celery task wrapper**

Create `sam2_worker/app/tasks.py`:
```python
"""SAM 2 tracking Celery task."""

import logging
import os
import traceback

from celery import Celery

from app.celery_app import app
from app.db import get_session
from app.models import Performance, PerformanceDancer, TrackingFrame

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
_celery_sender = Celery("sam2_dispatch", broker=REDIS_URL)

BATCH_SIZE = 500


def _make_cancel_checker(performance_id: int):
    import redis
    r = redis.from_url(REDIS_URL)
    def is_cancelled():
        return r.exists(f"cancel:{performance_id}") > 0
    return is_cancelled


@app.task(bind=True, name="sam2_worker.app.tasks.run_sam2_tracking")
def run_sam2_tracking(self, performance_id: int, video_path: str,
                      start_timestamp_ms: int, click_prompts: list,
                      selected_tracks: list):
    logger.info(f"SAM 2 tracking starting for performance {performance_id}")

    try:
        # Update status
        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "tracking"
                perf.pipeline_progress = {"stage": "tracking", "pct": 0.0,
                                          "message": "Loading SAM 2 model..."}

        is_cancelled = _make_cancel_checker(performance_id)

        def progress_callback(current, total):
            pct = (current / max(total, 1)) * 100
            with get_session() as session:
                perf = session.query(Performance).filter(Performance.id == performance_id).first()
                if perf:
                    perf.pipeline_progress = {
                        "stage": "tracking", "pct": round(pct, 1),
                        "message": f"Tracking dancers... frame {current}/{total}",
                    }

        # Run SAM 2 inference
        from app.inference import run_sam2_tracking as _run_tracking
        results = _run_tracking(
            video_path, start_timestamp_ms, click_prompts,
            is_cancelled=is_cancelled,
            progress_callback=progress_callback,
        )

        # Store tracking frames in batches
        with get_session() as session:
            batch = []
            for r in results:
                batch.append(TrackingFrame(
                    performance_id=performance_id,
                    dancer_index=r["dancer_index"],
                    timestamp_ms=r["timestamp_ms"],
                    bbox=r["bbox"],
                    mask_iou=r["mask_iou"],
                ))
                if len(batch) >= BATCH_SIZE:
                    session.add_all(batch)
                    session.flush()
                    batch = []
            if batch:
                session.add_all(batch)

        logger.info(f"Stored {len(results)} tracking frames for performance {performance_id}")

        # Dispatch pose pipeline
        _celery_sender.send_task(
            "worker.app.tasks.video_pipeline.run_pipeline",
            args=[performance_id, video_path],
            kwargs={"selected_tracks": selected_tracks},
            queue="video_pipeline",
        )
        logger.info(f"Dispatched pose pipeline for performance {performance_id}")

        return {"performance_id": performance_id, "status": "tracked",
                "tracking_frames": len(results)}

    except Exception as e:
        logger.error(f"SAM 2 tracking failed for performance {performance_id}: "
                     f"{e}\n{traceback.format_exc()}")
        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "failed"
                perf.error = f"SAM 2 tracking failed: {str(e)[:1000]}"
```

- [ ] **Step 3: Download SAM 2 model checkpoint**

```bash
mkdir -p ~/bharatanatyam-data/sam2
# Download from Meta's official release
wget -O ~/bharatanatyam-data/sam2/sam2.1_hiera_base_plus.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
```

- [ ] **Step 4: Rebuild and test SAM 2 worker**

```bash
podman-compose build sam2_worker
podman-compose up -d sam2_worker
sleep 5
podman exec source1_sam2_worker_1 python -c "
from sam2.build_sam import build_sam2_video_predictor
print('SAM 2 import OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add sam2_worker/app/inference.py sam2_worker/app/tasks.py
git commit -m "Add SAM 2 inference and tracking task"
```

---

## Phase 3: Main Worker — Cropped Pose Estimation

### Task 5: Replace tracker-based pose with crop-based pose

**Files:**
- Modify: `worker/app/pipeline/pose.py`
- Modify: `worker/app/pipeline/pose_config.py`
- Modify: `worker/app/tasks/video_pipeline.py`

- [ ] **Step 1: Add SAM2_FRAME_SKIP to pose_config.py**

In `worker/app/pipeline/pose_config.py`, add:
```python
SAM2_FRAME_SKIP = max(1, int(os.getenv("SAM2_FRAME_SKIP", "2")))
```

- [ ] **Step 2: Add `run_pose_estimation_cropped()` to pose.py**

Add this new function. It reads pre-computed bboxes from tracking_frames and runs RTMPose on cropped regions:

```python
def run_pose_estimation_cropped(
    video_path: str,
    metadata: dict,
    dancer_bboxes: list[dict],
    start_ms: int = 0,
    progress_callback=None,
    is_cancelled=None,
):
    """Yield per-frame pose dicts from RTMPose on pre-cropped dancer regions.

    Args:
        dancer_bboxes: List of {timestamp_ms, bbox, mask_iou} from tracking_frames.
        start_ms: Video timestamp to start from.

    Yields dicts with dancer_pose, left_hand, right_hand, face, timestamp_ms, bbox.
    """
    model = _init_model()

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    codec = metadata.get("codec", "h264")

    # Build lookup: timestamp_ms -> bbox dict
    bbox_lookup = {b["timestamp_ms"]: b for b in dancer_bboxes}
    timestamps = sorted(bbox_lookup.keys())
    total = len(timestamps)

    if total == 0:
        return

    # Build ffmpeg command with output-mode seeking for frame accuracy
    start_sec = start_ms / 1000.0
    frame_skip = POSE_FRAME_SKIP
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    nvdec = NVDEC_CODECS.get(codec)
    if nvdec:
        cmd += ["-hwaccel", "cuda", "-c:v", nvdec]

    cmd += ["-i", video_path, "-ss", str(start_sec)]

    vf_parts = []
    if frame_skip > 1:
        vf_parts.append(f"select='not(mod(n\\,{frame_skip}))'")
        vf_parts.append("setpts=N/FRAME_RATE/TB")
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-v", "error", "pipe:1"]

    frame_size = width * height * 3
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_idx = 0
    BBOX_PAD = 0.25  # 25% padding around bbox

    try:
        while True:
            raw = process.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break

            timestamp_ms = start_ms + int(frame_idx * frame_skip / fps * 1000)

            # Find tracking bbox for this timestamp
            entry = bbox_lookup.get(timestamp_ms)
            if entry is None:
                # Try nearest timestamp within 50ms
                nearest = min(timestamps, key=lambda t: abs(t - timestamp_ms), default=None)
                if nearest is not None and abs(nearest - timestamp_ms) < 50:
                    entry = bbox_lookup[nearest]

            frame_idx += 1

            if entry is None or entry.get("mask_iou", 0) < 0.3:
                continue

            bbox = entry["bbox"]
            bw = bbox["x_max"] - bbox["x_min"]
            bh = bbox["y_max"] - bbox["y_min"]
            if bw < 0.01 or bh < 0.01:
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            # Denormalize bbox to pixels and add padding
            pad_x = bw * BBOX_PAD
            pad_y = bh * BBOX_PAD
            px1 = max(0, int((bbox["x_min"] - pad_x) * width))
            py1 = max(0, int((bbox["y_min"] - pad_y) * height))
            px2 = min(width, int((bbox["x_max"] + pad_x) * width))
            py2 = min(height, int((bbox["y_max"] + pad_y) * height))

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            crop_h, crop_w = crop.shape[:2]

            # Run RTMPose on crop
            keypoints, scores = model(crop)
            if keypoints is None or len(keypoints) == 0:
                continue

            # Extract single person (largest) from crop
            pose_data = _extract_pose_data(keypoints, scores, crop_w, crop_h)

            # Transform normalized crop coords to full-frame normalized coords
            for kp_name, kp in pose_data.get("dancer_pose", {}).items():
                kp["x"] = kp["x"] * (crop_w / width) + (px1 / width)
                kp["y"] = kp["y"] * (crop_h / height) + (py1 / height)

            # Transform hands
            for hand_key in ("left_hand", "right_hand"):
                for kp_name, kp in pose_data.get(hand_key, {}).items():
                    kp["x"] = kp["x"] * (crop_w / width) + (px1 / width)
                    kp["y"] = kp["y"] * (crop_h / height) + (py1 / height)

            # Transform face
            for kp in pose_data.get("face", []):
                kp["x"] = kp["x"] * (crop_w / width) + (px1 / width)
                kp["y"] = kp["y"] * (crop_h / height) + (py1 / height)

            # Compute full-frame bbox from transformed keypoints
            pose_data["bbox"] = (
                round(bbox["x_min"], 5), round(bbox["y_min"], 5),
                round(bbox["x_max"], 5), round(bbox["y_max"], 5),
            )
            pose_data["timestamp_ms"] = timestamp_ms
            yield pose_data

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total)
            if is_cancelled and frame_idx % 10 == 0 and is_cancelled():
                break

    finally:
        process.stdout.close()
        process.wait()
```

- [ ] **Step 3: Modify video_pipeline.py multi-dancer branch**

Replace the tracker-based approach with tracking_frames-based approach:

In the `if selected_tracks:` branch, replace the seed_bboxes/tracker section with:
```python
if selected_tracks:
    track_id_to_info = {t["track_id"]: t for t in selected_tracks}
    selected_ids = set(track_id_to_info.keys())

    # Read SAM 2 tracking bboxes from DB
    start_ms = 0
    with get_session() as session:
        perf = session.query(Performance).filter(Performance.id == performance_id).first()
        if perf and perf.start_timestamp_ms:
            start_ms = perf.start_timestamp_ms

    # Adjust effective duration
    effective_duration_ms = metadata["duration_ms"] - start_ms if start_ms else metadata["duration_ms"]

    # Process each dancer independently
    per_dancer_frames = {}
    for track_id in selected_ids:
        info = track_id_to_info[track_id]
        dancer_index = track_id

        # Fetch tracking bboxes for this dancer
        with get_session() as session:
            from app.models.performance import TrackingFrame
            rows = session.query(TrackingFrame).filter(
                TrackingFrame.performance_id == performance_id,
                TrackingFrame.dancer_index == dancer_index,
            ).order_by(TrackingFrame.timestamp_ms).all()
            dancer_bboxes = [
                {"timestamp_ms": r.timestamp_ms, "bbox": r.bbox, "mask_iou": r.mask_iou}
                for r in rows
            ]

        if not dancer_bboxes:
            logger.warning(f"No tracking data for dancer {dancer_index}")
            per_dancer_frames[track_id] = []
            continue

        # Run pose estimation on cropped regions
        frames_data = []
        frame_gen = run_pose_estimation_cropped(
            video_path, metadata, dancer_bboxes, start_ms=start_ms,
            progress_callback=pose_progress, is_cancelled=is_cancelled,
        )
        for fd in frame_gen:
            frames_data.append(fd)
        per_dancer_frames[track_id] = frames_data
```

The rest of the pipeline (store frames, compute scores, LLM coaching) remains the same.

**Performance note:** This processes each dancer sequentially (separate ffmpeg pass per dancer). For 3 dancers on a 289s video, that's 3 video reads. This is a known limitation — a future optimization could read the video once and crop all dancers per frame. For now, sequential is simpler and correctness matters more than speed.

- [ ] **Step 4: Pass start_ms to beat detection**

In `video_pipeline.py`, modify the beat analysis call:
```python
beat_data = run_beat_analysis(video_path, metadata, start_ms=start_ms)
```

In `beat_detection.py`, add `start_ms` parameter to `run_beat_analysis()` — use ffmpeg's `-ss` for audio extraction starting from that timestamp.

- [ ] **Step 5: Verify worker builds**

```bash
podman-compose build worker
```

- [ ] **Step 6: Commit**

```bash
git add worker/app/pipeline/pose.py worker/app/pipeline/pose_config.py worker/app/tasks/video_pipeline.py
git commit -m "Add cropped pose estimation using SAM 2 tracking bboxes"
```

---

## Phase 4: Frontend — Click-to-Select UI

### Task 6: SelectFrame page — video player with click-to-select

**Files:**
- Create: `frontend/src/pages/SelectFrame.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/api/performances.ts`

- [ ] **Step 1: Add API functions**

In `frontend/src/api/performances.ts`, add:
```typescript
export interface ClickPrompt {
  x: number;
  y: number;
  label?: string;
}

export async function selectFrame(
  id: number,
  startTimestampMs: number,
  prompts: ClickPrompt[],
): Promise<{ status: string; dancers_selected: number }> {
  const { data } = await api.post(`/performances/${id}/select-frame`, {
    start_timestamp_ms: startTimestampMs,
    prompts,
  });
  return data;
}

export async function resetTracking(id: number): Promise<{ status: string }> {
  const { data } = await api.post(`/performances/${id}/reset-tracking`);
  return data;
}
```

- [ ] **Step 2: Create SelectFrame.tsx**

Create `frontend/src/pages/SelectFrame.tsx`. This page shows:
- A `<video>` element with the uploaded video (scrub bar for seeking)
- A `<canvas>` overlay for click interaction
- Colored dots at click points, list of selected dancers below
- "Analyze from here" button

Key implementation details:
- Video source: `/uploads/{video_key}` (same as VideoReview)
- Click handler: `canvas.addEventListener("click", ...)` → normalize coords to `offsetX / canvas.width` (matching video natural dimensions)
- Pause video on click (user is selecting a frame)
- Get current time: `video.currentTime * 1000` for `start_timestamp_ms`
- On submit: call `selectFrame(id, timestamp, prompts)` → navigate to `/processing/${id}`

Use the same DANCER_COLORS array from DancerSelection.tsx.

- [ ] **Step 3: Update App.tsx routing**

Replace:
```typescript
<Route path="/select-dancers/:performanceId" element={<DancerSelection />} />
```
With:
```typescript
<Route path="/select-frame/:performanceId" element={<SelectFrame />} />
```

Update imports accordingly.

- [ ] **Step 4: Verify frontend builds**

```bash
podman exec source1_frontend_1 npx tsc --noEmit 2>&1 | grep -v "__tests__"
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/SelectFrame.tsx frontend/src/App.tsx frontend/src/api/performances.ts
git commit -m "Add SelectFrame page with click-to-select dancer UI"
```

### Task 7: Update Dashboard and ProcessingStatus for new statuses

**Files:**
- Modify: `frontend/src/pages/Dashboard.tsx`
- Modify: `frontend/src/pages/ProcessingStatus.tsx`

- [ ] **Step 1: Update Dashboard.tsx**

Change the upload handler to navigate to `/select-frame/` instead of `/processing/`. Update the performance list to handle new statuses:
- `transcoding` → spinner + "Preparing video..."
- `uploaded` → link to `/select-frame/${id}`
- `tracking` → spinner + "Tracking dancers..."

- [ ] **Step 2: Update ProcessingStatus.tsx**

Update the STAGES array to include `tracking`:
```typescript
const STAGES = [
  { key: "tracking", label: "Tracking Dancers", weight: 30 },
  { key: "ingest", label: "Ingest Video", weight: 3 },
  { key: "beat_detection", label: "Beat Detection", weight: 4 },
  { key: "pose_estimation", label: "Pose Estimation", weight: 35 },
  { key: "pose_analysis", label: "Pose Analysis", weight: 5 },
  { key: "llm_synthesis", label: "AI Coaching", weight: 10 },
  { key: "scoring", label: "Scoring", weight: 3 },
  { key: "complete", label: "Complete", weight: 10 },
];
```

Update the redirect logic:
- `uploaded` → redirect to `/select-frame/${id}`
- Remove `awaiting_selection` redirect

Update the stop button to also show for `tracking` status.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/Dashboard.tsx frontend/src/pages/ProcessingStatus.tsx
git commit -m "Update Dashboard and ProcessingStatus for SAM 2 tracking flow"
```

---

## Phase 5: Cleanup & End-to-End Test

### Task 8: Remove old tracker code and detection pass

**Files:**
- Delete: `worker/app/pipeline/tracker.py`
- Delete: `worker/app/pipeline/bytetrack.py`
- Delete: `worker/app/pipeline/reid.py`
- Delete: `worker/app/pipeline/appearance.py`
- Delete: `worker/export_reid_model.py`
- Delete: `worker/app/tasks/detect_dancers.py`
- Delete: `frontend/src/pages/DancerSelection.tsx`
- Modify: `worker/Dockerfile` (remove Re-ID model export)
- Modify: `worker/app/pipeline/pose.py` (remove `run_pose_estimation_multi`, `run_detection_pass`)

- [ ] **Step 1: Remove deleted files**

```bash
git rm worker/app/pipeline/tracker.py
git rm worker/app/pipeline/bytetrack.py
git rm worker/app/pipeline/reid.py
git rm worker/app/pipeline/appearance.py
git rm worker/export_reid_model.py
git rm worker/app/tasks/detect_dancers.py
git rm frontend/src/pages/DancerSelection.tsx
```

- [ ] **Step 2: Clean up pose.py**

Remove `run_pose_estimation_multi()`, `run_detection_pass()`, `_extract_motion_state()`, and all tracker/identity imports.

- [ ] **Step 3: Clean up worker/Dockerfile**

Remove the `python export_reid_model.py` line from the Dockerfile. Remove `onnx` from requirements.txt (keep scipy for librosa).

- [ ] **Step 4: Clean up pose_config.py**

Remove tracker-related config vars (`TRACKER_*`, `REID_*`).

- [ ] **Step 5: Verify builds**

```bash
podman-compose build worker
podman-compose build sam2_worker
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Remove old tracker, detection pass, and identity recovery code (-2400 lines)"
```

### Task 9: End-to-end test

- [ ] **Step 1: Start all services**

```bash
podman-compose up -d
sleep 10
podman logs source1_sam2_worker_1 2>&1 | tail -5
podman logs source1_worker_1 2>&1 | tail -5
```

- [ ] **Step 2: Upload test video via frontend**

Open `http://localhost:5173`, upload `occlusion_test_h264.mp4`. Verify redirect to `/select-frame/:id`.

- [ ] **Step 3: Select dancers**

Scrub to ~5s where all 3 standing dancers are visible. Click on each dancer. Label them. Click "Analyze from here."

- [ ] **Step 4: Monitor tracking**

Watch ProcessingStatus page. Verify SAM 2 tracking progress shows and completes.

```bash
podman logs --follow source1_sam2_worker_1 2>&1 | grep -E "tracking|SAM|prompt|frame"
```

- [ ] **Step 5: Monitor pose estimation**

After SAM 2 completes, verify pose pipeline is dispatched and runs.

```bash
podman logs --follow source1_worker_1 2>&1 | grep -E "pipeline|pose|frame"
```

- [ ] **Step 6: Verify review page**

Once complete, check the review page:
- All 3 dancers have skeletons rendered correctly
- Skeletons are positioned on the correct dancers (no swapping)
- V-formation transition at ~31s doesn't cause tracking loss

- [ ] **Step 7: Commit and push**

```bash
git add -A
git commit -m "SAM 2 tracking integration complete — end-to-end tested"
git push
```

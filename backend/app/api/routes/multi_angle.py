import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models.dancer import Dancer
from app.models.performance import Performance, MultiAngleGroup, PerformanceDancer
from app.models.analysis import Analysis, MultiAngleAnalysis
from app.tasks import dispatch_detection, dispatch_multi_angle_pipeline
from app.schemas.performance import (
    MultiAngleUploadResponse,
    MultiAngleGroupResponse,
    MultiAngleGroupListItem,
    CrossViewDancerLinkRequest,
    UploadResponse,
)

router = APIRouter(prefix="/api/multi-angle", tags=["multi-angle"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@router.post("/upload", response_model=MultiAngleUploadResponse)
async def upload_multi_angle(
    files: list[UploadFile] = File(...),
    dancer_id: int = Form(...),
    camera_labels: str = Form(...),  # comma-separated: "Front,Side"
    item_name: str = Form(None),
    item_type: str = Form(None),
    talam: str = Form(None),
    ragam: str = Form(None),
    db: Session = Depends(get_db),
):
    """Upload two or more videos as a multi-angle group."""
    dancer = db.query(Dancer).filter(Dancer.id == dancer_id).first()
    if not dancer:
        raise HTTPException(status_code=404, detail="Dancer not found")

    labels = [l.strip() for l in camera_labels.split(",")]
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 video files required for multi-angle analysis")
    if len(labels) != len(files):
        raise HTTPException(status_code=400, detail=f"Number of camera labels ({len(labels)}) must match number of files ({len(files)})")

    # Create the group
    group = MultiAngleGroup(
        dancer_id=dancer_id,
        item_name=item_name,
        item_type=item_type,
        talam=talam,
        ragam=ragam,
        status="uploading",
    )
    db.add(group)
    db.flush()

    results = []
    for file, label in zip(files, labels):
        filename = file.filename or "video.mp4"
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported for {filename}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        video_key = f"{uuid.uuid4()}{ext}"
        video_path = f"/app/uploads/{video_key}"

        with open(video_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        performance = Performance(
            dancer_id=dancer_id,
            multi_angle_group_id=group.id,
            camera_label=label,
            item_name=item_name,
            item_type=item_type,
            talam=talam,
            ragam=ragam,
            video_url=f"/uploads/{video_key}",
            video_key=video_key,
            status="queued",
        )
        db.add(performance)
        db.flush()

        task_id = dispatch_detection(performance.id, video_path)
        performance.task_id = task_id

        results.append(UploadResponse(
            performance_id=performance.id,
            task_id=task_id,
            status="queued",
        ))

    group.status = "detecting"
    db.commit()

    return MultiAngleUploadResponse(group_id=group.id, performances=results)


@router.get("/groups", response_model=list[MultiAngleGroupListItem])
def list_groups(dancer_id: int | None = None, db: Session = Depends(get_db)):
    query = db.query(MultiAngleGroup).order_by(MultiAngleGroup.created_at.desc())
    if dancer_id is not None:
        query = query.filter(MultiAngleGroup.dancer_id == dancer_id)
    groups = query.all()

    results = []
    for g in groups:
        perf_count = db.query(Performance).filter(Performance.multi_angle_group_id == g.id).count()
        best_score = db.query(MultiAngleAnalysis.overall_score).filter(
            MultiAngleAnalysis.multi_angle_group_id == g.id,
            MultiAngleAnalysis.overall_score.isnot(None),
        ).order_by(MultiAngleAnalysis.overall_score.desc()).first()

        item = MultiAngleGroupListItem.model_validate(g)
        item.performance_count = perf_count
        item.overall_score = best_score[0] if best_score else None
        results.append(item)

    return results


@router.get("/groups/{group_id}", response_model=MultiAngleGroupResponse)
def get_group(group_id: int, db: Session = Depends(get_db)):
    group = (
        db.query(MultiAngleGroup)
        .options(
            joinedload(MultiAngleGroup.performances).joinedload(Performance.analysis),
            joinedload(MultiAngleGroup.performances).joinedload(Performance.detected_persons),
            joinedload(MultiAngleGroup.performances).joinedload(Performance.performance_dancers),
            joinedload(MultiAngleGroup.multi_angle_analyses),
        )
        .filter(MultiAngleGroup.id == group_id)
        .first()
    )
    if not group:
        raise HTTPException(status_code=404, detail="Multi-angle group not found")
    return group


@router.get("/groups/{group_id}/status")
def get_group_status(group_id: int, db: Session = Depends(get_db)):
    """Poll status of all performances in the group."""
    group = db.query(MultiAngleGroup).filter(MultiAngleGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Multi-angle group not found")

    performances = db.query(Performance).filter(Performance.multi_angle_group_id == group_id).all()

    perf_statuses = []
    all_awaiting = True
    any_failed = False
    all_complete = True

    for p in performances:
        perf_statuses.append({
            "performance_id": p.id,
            "camera_label": p.camera_label,
            "status": p.status,
            "pipeline_progress": p.pipeline_progress,
        })
        if p.status != "awaiting_selection":
            all_awaiting = False
        if p.status == "failed":
            any_failed = True
        if p.status != "complete":
            all_complete = False

    return {
        "group_id": group_id,
        "group_status": group.status,
        "all_awaiting_selection": all_awaiting,
        "any_failed": any_failed,
        "all_complete": all_complete,
        "performances": perf_statuses,
    }


@router.post("/groups/{group_id}/link-dancers")
def link_dancers_across_views(
    group_id: int,
    body: CrossViewDancerLinkRequest,
    db: Session = Depends(get_db),
):
    """Link dancers across views and start the multi-angle pipeline.

    After dancer selection on each individual performance, the user links
    dancers across views (e.g., "Priya" is track 0 in front view, track 1 in side view).
    This creates PerformanceDancer records for each view and dispatches the pipeline.
    """
    group = db.query(MultiAngleGroup).filter(MultiAngleGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Multi-angle group not found")

    performances = db.query(Performance).filter(Performance.multi_angle_group_id == group_id).all()
    perf_map = {p.id: p for p in performances}

    if not body.links:
        raise HTTPException(status_code=400, detail="At least one dancer link is required")

    # Validate all referenced performances belong to this group
    for link in body.links:
        for perf_id in link.performance_tracks:
            if perf_id not in perf_map:
                raise HTTPException(
                    status_code=400,
                    detail=f"Performance {perf_id} does not belong to group {group_id}",
                )

    # Create PerformanceDancer records for each view
    all_selected_tracks = {}  # performance_id -> list of {track_id, performance_dancer_id, label}
    for link in body.links:
        for perf_id, track_id in link.performance_tracks.items():
            pd = PerformanceDancer(
                performance_id=perf_id,
                track_id=track_id,
                label=link.label,
            )
            db.add(pd)
            db.flush()

            if perf_id not in all_selected_tracks:
                all_selected_tracks[perf_id] = []
            all_selected_tracks[perf_id].append({
                "track_id": track_id,
                "performance_dancer_id": pd.id,
                "label": link.label,
            })

    # Dispatch individual pipelines for each performance
    for perf_id, selected_tracks in all_selected_tracks.items():
        perf = perf_map[perf_id]
        video_path = f"/app/uploads/{perf.video_key}"
        from app.tasks import dispatch_pipeline
        task_id = dispatch_pipeline(perf_id, video_path, selected_tracks=selected_tracks)
        perf.task_id = task_id
        perf.status = "queued"
        perf.pipeline_progress = {"stage": "queued", "pct": 0.0}

    group.status = "processing"
    db.commit()

    # Dispatch multi-angle fusion (will wait for individual pipelines)
    fusion_task_id = dispatch_multi_angle_pipeline(group_id)

    return {
        "status": "processing",
        "fusion_task_id": fusion_task_id,
        "performances_dispatched": len(all_selected_tracks),
        "dancers_linked": len(body.links),
    }


@router.delete("/groups/{group_id}")
def delete_group(group_id: int, db: Session = Depends(get_db)):
    group = db.query(MultiAngleGroup).filter(MultiAngleGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Multi-angle group not found")

    # Delete associated performances (cascade will handle frames, etc.)
    performances = db.query(Performance).filter(Performance.multi_angle_group_id == group_id).all()
    for perf in performances:
        if perf.task_id and perf.status in ("queued", "processing"):
            from app.tasks import celery_app
            celery_app.control.revoke(perf.task_id, terminate=True)
        if perf.video_key:
            import os
            video_path = f"/app/uploads/{perf.video_key}"
            if os.path.exists(video_path):
                os.remove(video_path)
        db.delete(perf)

    db.delete(group)
    db.commit()
    return {"status": "deleted"}

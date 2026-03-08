from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models.performance import Performance, DetectedPerson, PerformanceDancer
from app.tasks import dispatch_pipeline

from app.models.analysis import Analysis
from app.schemas.performance import (
    PerformanceResponse,
    PerformanceStatusResponse,
    PerformanceListItem,
    FrameResponse,
    DetectedPersonResponse,
    DancerSelectionRequest,
)

router = APIRouter(prefix="/api/performances", tags=["performances"])


@router.get("/", response_model=list[PerformanceListItem])
def list_performances(dancer_id: int | None = None, db: Session = Depends(get_db)):
    query = db.query(Performance).order_by(Performance.created_at.desc())
    if dancer_id is not None:
        query = query.filter(Performance.dancer_id == dancer_id)
    performances = query.all()

    results = []
    for perf in performances:
        # Get overall score from first analysis (or highest among dancers)
        best_score = db.query(Analysis.overall_score).filter(
            Analysis.performance_id == perf.id,
            Analysis.overall_score.isnot(None),
        ).order_by(Analysis.overall_score.desc()).first()

        item = PerformanceListItem.model_validate(perf)
        item.overall_score = best_score[0] if best_score else None
        results.append(item)

    return results


@router.get("/{performance_id}", response_model=PerformanceResponse)
def get_performance(performance_id: int, db: Session = Depends(get_db)):
    performance = (
        db.query(Performance)
        .options(
            joinedload(Performance.analysis),
            joinedload(Performance.detected_persons),
            joinedload(Performance.performance_dancers),
        )
        .filter(Performance.id == performance_id)
        .first()
    )
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    # Frames are loaded separately via /frames endpoint for performance
    return performance


@router.get("/{performance_id}/frames", response_model=list[FrameResponse])
def get_performance_frames(performance_id: int, db: Session = Depends(get_db)):
    """Return all frames for a performance. Loaded separately for performance."""
    from app.models.analysis import Frame
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    frames = (
        db.query(Frame)
        .filter(Frame.performance_id == performance_id)
        .order_by(Frame.timestamp_ms)
        .all()
    )
    return frames


@router.get("/{performance_id}/status", response_model=PerformanceStatusResponse)
def get_performance_status(performance_id: int, db: Session = Depends(get_db)):
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    return performance


@router.get("/{performance_id}/detected-persons", response_model=list[DetectedPersonResponse])
def get_detected_persons(performance_id: int, db: Session = Depends(get_db)):
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    return db.query(DetectedPerson).filter(DetectedPerson.performance_id == performance_id).order_by(DetectedPerson.area.desc()).all()


@router.post("/{performance_id}/select-dancers")
def select_dancers(performance_id: int, body: DancerSelectionRequest, db: Session = Depends(get_db)):
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    if performance.status != "awaiting_selection":
        raise HTTPException(status_code=400, detail=f"Performance is not awaiting selection (status: {performance.status})")
    if not body.selections:
        raise HTTPException(status_code=400, detail="At least one dancer must be selected")

    # Create PerformanceDancer records
    selected_tracks = []
    for sel in body.selections:
        pd = PerformanceDancer(
            performance_id=performance_id,
            track_id=sel.track_id,
            label=sel.label,
        )
        db.add(pd)
        db.flush()
        selected_tracks.append({"track_id": sel.track_id, "performance_dancer_id": pd.id, "label": sel.label})

    # Dispatch full pipeline with selected dancers
    video_path = f"/app/uploads/{performance.video_key}"
    task_id = dispatch_pipeline(performance_id, video_path, selected_tracks=selected_tracks)
    performance.task_id = task_id
    performance.status = "queued"
    performance.pipeline_progress = {"stage": "queued", "pct": 0.0}
    db.commit()

    return {"task_id": task_id, "status": "queued", "dancers_selected": len(selected_tracks)}


@router.post("/{performance_id}/stop")
def stop_performance(performance_id: int, db: Session = Depends(get_db)):
    """Signal the worker to stop pose estimation early and proceed to analysis."""
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    if performance.status not in ("queued", "processing", "detecting"):
        raise HTTPException(status_code=400, detail=f"Performance is not processing (status: {performance.status})")

    # Set cancellation flag in Redis so the worker stops pose estimation early
    # The pipeline will still run analysis/scoring on collected frames
    import redis
    from app.config import settings
    r = redis.from_url(settings.redis_url)
    r.set(f"cancel:{performance_id}", "1", ex=300)

    db.commit()

    return {"status": "stopping", "message": "Stopping pose estimation, analysis will run on collected frames"}


@router.delete("/{performance_id}")
def delete_performance(performance_id: int, db: Session = Depends(get_db)):
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")

    # Revoke celery task if still processing
    if performance.task_id and performance.status in ("queued", "processing"):
        from app.tasks import celery_app
        celery_app.control.revoke(performance.task_id, terminate=True)

    # Delete video file
    if performance.video_key:
        import os
        video_path = f"/app/uploads/{performance.video_key}"
        if os.path.exists(video_path):
            os.remove(video_path)

    db.delete(performance)
    db.commit()
    return {"status": "deleted"}

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
    TimelineFrameResponse,
    DetectedPersonResponse,
    DancerSelectionRequest,
    SelectFrameRequest,
)

router = APIRouter(prefix="/api/performances", tags=["performances"])


@router.get("/", response_model=list[PerformanceListItem])
def list_performances(dancer_id: int | None = None, db: Session = Depends(get_db)):
    from sqlalchemy import func

    # Single query with subquery for best score (avoids N+1)
    best_score_sq = (
        db.query(
            Analysis.performance_id,
            func.max(Analysis.overall_score).label("best_score"),
        )
        .filter(Analysis.overall_score.isnot(None))
        .group_by(Analysis.performance_id)
        .subquery()
    )

    query = (
        db.query(Performance, best_score_sq.c.best_score)
        .outerjoin(best_score_sq, Performance.id == best_score_sq.c.performance_id)
        .order_by(Performance.created_at.desc())
    )
    if dancer_id is not None:
        query = query.filter(Performance.dancer_id == dancer_id)

    results = []
    for perf, best_score in query.all():
        item = PerformanceListItem.model_validate(perf)
        item.overall_score = best_score
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
    # Check if any frame has real 3D data (not JSON null)
    from app.models.analysis import Frame
    from sqlalchemy import func, cast, String
    has_3d = db.query(Frame.id).filter(
        Frame.performance_id == performance_id,
        Frame.joints_3d.isnot(None),
        cast(Frame.joints_3d, String) != "null",
    ).limit(1).first() is not None
    performance.has_3d = has_3d
    return performance


@router.get("/{performance_id}/frames", response_model=list[FrameResponse])
def get_performance_frames(performance_id: int, include_3d: bool = False, db: Session = Depends(get_db)):
    """Return all frames for a performance. Only loads columns needed for display.

    Pass include_3d=true to include joints_3d/world_position/foot_contact (large payload).
    """
    from app.models.analysis import Frame
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    # Only select columns in FrameResponse — skip heavy left_hand, right_hand, face JSON
    columns = [Frame.id, Frame.timestamp_ms, Frame.dancer_pose, Frame.performance_dancer_id]
    if include_3d:
        columns.extend([Frame.joints_3d, Frame.world_position, Frame.foot_contact])
    rows = (
        db.query(*columns)
        .filter(Frame.performance_id == performance_id)
        .order_by(Frame.timestamp_ms)
        .all()
    )
    return [
        FrameResponse(
            id=r.id, timestamp_ms=r.timestamp_ms, dancer_pose=r.dancer_pose,
            performance_dancer_id=r.performance_dancer_id,
            joints_3d=r.joints_3d if include_3d else None,
            world_position=r.world_position if include_3d else None,
            foot_contact=r.foot_contact if include_3d else None,
        )
        for r in rows
    ]


@router.get("/{performance_id}/timeline", response_model=list[TimelineFrameResponse])
def get_performance_timeline(performance_id: int, db: Session = Depends(get_db)):
    """Return per-frame angle and balance metrics for timeline visualization."""
    from app.models.analysis import Frame, JointAngleState, BalanceMetrics

    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")

    rows = (
        db.query(
            Frame.timestamp_ms,
            Frame.performance_dancer_id,
            Frame.foot_contact,
            JointAngleState.aramandi_angle,
            JointAngleState.torso_uprightness,
            JointAngleState.arm_extension_left,
            JointAngleState.arm_extension_right,
            JointAngleState.hip_symmetry,
            JointAngleState.knee_angle_3d,
            JointAngleState.torso_angle_3d,
            JointAngleState.torso_twist,
            BalanceMetrics.stability_score,
        )
        .outerjoin(JointAngleState, JointAngleState.frame_id == Frame.id)
        .outerjoin(BalanceMetrics, BalanceMetrics.frame_id == Frame.id)
        .filter(Frame.performance_id == performance_id)
        .order_by(Frame.timestamp_ms)
        .all()
    )

    return [
        TimelineFrameResponse(
            timestamp_ms=r.timestamp_ms,
            performance_dancer_id=r.performance_dancer_id,
            aramandi_angle=r.aramandi_angle,
            torso_uprightness=r.torso_uprightness,
            arm_extension_left=r.arm_extension_left,
            arm_extension_right=r.arm_extension_right,
            hip_symmetry=r.hip_symmetry,
            stability_score=r.stability_score,
            knee_angle_3d=r.knee_angle_3d,
            torso_angle_3d=r.torso_angle_3d,
            torso_twist=r.torso_twist,
            foot_contact_left=(r.foot_contact or {}).get("left_heel"),
            foot_contact_right=(r.foot_contact or {}).get("right_heel"),
        )
        for r in rows
    ]


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
    if performance.status not in ("queued", "processing", "detecting", "tracking", "transcoding"):
        raise HTTPException(status_code=400, detail=f"Performance is not processing (status: {performance.status})")

    # Set cancellation flag in Redis so the worker stops pose estimation early
    # The pipeline will still run analysis/scoring on collected frames
    import redis
    from app.config import settings
    r = redis.from_url(settings.redis_url)
    r.set(f"cancel:{performance_id}", "1", ex=300)

    db.commit()

    return {"status": "stopping", "message": "Stopping pose estimation, analysis will run on collected frames"}


@router.post("/{performance_id}/select-frame")
def select_frame(performance_id: int, body: SelectFrameRequest, db: Session = Depends(get_db)):
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    if performance.status != "uploaded":
        raise HTTPException(status_code=400, detail=f"Performance not ready (status: {performance.status})")
    if not body.prompts:
        raise HTTPException(status_code=400, detail="At least one dancer must be selected")

    performance.start_timestamp_ms = body.start_timestamp_ms
    performance.click_prompts = [p.model_dump() for p in body.prompts]

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

    # Commit DB FIRST, then dispatch (prevents race if dispatch fails)
    performance.status = "tracking"
    db.commit()

    video_path = f"/app/uploads/{performance.video_key}"
    from app.tasks import dispatch_sam2
    dispatch_sam2(performance_id, video_path, body.start_timestamp_ms,
                  [p.model_dump() for p in body.prompts], selected_tracks)

    return {"status": "tracking", "dancers_selected": len(body.prompts)}


@router.post("/{performance_id}/retry")
def retry_performance(performance_id: int, db: Session = Depends(get_db)):
    """Retry a failed performance, resuming from the last completed stage."""
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    if performance.status != "failed":
        raise HTTPException(status_code=400, detail=f"Can only retry failed performances (status: {performance.status})")

    # Build selected_tracks from existing PerformanceDancer rows
    dancers = db.query(PerformanceDancer).filter(
        PerformanceDancer.performance_id == performance_id,
    ).all()
    selected_tracks = [
        {"track_id": d.track_id, "performance_dancer_id": d.id, "label": d.label}
        for d in dancers
    ] if dancers else None

    performance.status = "processing"
    performance.error = None
    db.commit()

    video_path = f"/app/uploads/{performance.video_key}"
    from app.tasks import dispatch_pipeline
    dispatch_pipeline(performance_id, video_path, selected_tracks, resume=True)

    return {"status": "processing", "message": "Retrying from last completed stage"}


@router.post("/{performance_id}/reset-tracking")
def reset_tracking(performance_id: int, db: Session = Depends(get_db)):
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    if performance.status not in ("uploaded", "tracking", "failed"):
        raise HTTPException(status_code=400, detail=f"Cannot reset (status: {performance.status})")

    if performance.status == "tracking":
        import redis as r
        import os
        redis_client = r.from_url(os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"))
        redis_client.set(f"cancel:{performance_id}", "1", ex=300)

    # Delete all related data
    from app.models.performance import TrackingFrame
    from app.models.analysis import Frame, JointAngleState, BalanceMetrics
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


@router.delete("/{performance_id}")
def delete_performance(performance_id: int, db: Session = Depends(get_db)):
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")

    # Revoke celery task if still processing
    if performance.task_id and performance.status in ("queued", "processing", "detecting"):
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

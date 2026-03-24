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
    # Frames are loaded separately via /frames endpoint for performance
    return performance


@router.get("/{performance_id}/frames", response_model=list[FrameResponse])
def get_performance_frames(performance_id: int, db: Session = Depends(get_db)):
    """Return all frames for a performance. Only loads columns needed for display."""
    from app.models.analysis import Frame
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    # Only select columns in FrameResponse — skip heavy left_hand, right_hand, face JSON
    rows = (
        db.query(
            Frame.id, Frame.timestamp_ms, Frame.dancer_pose,
            Frame.performance_dancer_id, Frame.joints_3d,
            Frame.world_position, Frame.foot_contact,
        )
        .filter(Frame.performance_id == performance_id)
        .order_by(Frame.timestamp_ms)
        .all()
    )
    return [
        FrameResponse(
            id=r.id, timestamp_ms=r.timestamp_ms, dancer_pose=r.dancer_pose,
            performance_dancer_id=r.performance_dancer_id, joints_3d=r.joints_3d,
            world_position=r.world_position, foot_contact=r.foot_contact,
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

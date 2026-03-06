from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models.performance import Performance

from app.schemas.performance import PerformanceResponse, PerformanceStatusResponse

router = APIRouter(prefix="/api/performances", tags=["performances"])


@router.get("/{performance_id}", response_model=PerformanceResponse)
def get_performance(performance_id: int, db: Session = Depends(get_db)):
    performance = (
        db.query(Performance)
        .options(joinedload(Performance.frames), joinedload(Performance.analysis))
        .filter(Performance.id == performance_id)
        .first()
    )
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    return performance


@router.get("/{performance_id}/status", response_model=PerformanceStatusResponse)
def get_performance_status(performance_id: int, db: Session = Depends(get_db)):
    performance = db.query(Performance).filter(Performance.id == performance_id).first()
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    return performance


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

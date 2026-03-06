import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dancer import Dancer
from app.models.performance import Performance
from app.tasks import dispatch_pipeline
from app.schemas.performance import UploadResponse

router = APIRouter(prefix="/api/upload", tags=["upload"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB


@router.post("/", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    dancer_id: int = Form(...),
    item_name: str = Form(None),
    item_type: str = Form(None),
    talam: str = Form(None),
    ragam: str = Form(None),
    db: Session = Depends(get_db),
):
    # Validate dancer exists
    dancer = db.query(Dancer).filter(Dancer.id == dancer_id).first()
    if not dancer:
        raise HTTPException(status_code=404, detail="Dancer not found")

    # Validate file extension
    filename = file.filename or "video.mp4"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save file
    video_key = f"{uuid.uuid4()}{ext}"
    video_path = f"/app/uploads/{video_key}"

    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    # Create performance record
    performance = Performance(
        dancer_id=dancer_id,
        item_name=item_name,
        item_type=item_type,
        talam=talam,
        ragam=ragam,
        video_url=f"/uploads/{video_key}",
        video_key=video_key,
        status="queued",
    )
    db.add(performance)
    db.commit()
    db.refresh(performance)

    # Dispatch pipeline
    task_id = dispatch_pipeline(performance.id, video_path)
    performance.task_id = task_id
    db.commit()

    return UploadResponse(
        performance_id=performance.id,
        task_id=task_id,
        status="queued",
    )

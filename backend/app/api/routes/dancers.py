import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dancer import Dancer
from app.schemas.dancer import DancerCreate, DancerResponse

router = APIRouter(prefix="/api/dancers", tags=["dancers"])


@router.post("/", response_model=DancerResponse)
def create_dancer(dancer: DancerCreate, db: Session = Depends(get_db)):
    existing = db.query(Dancer).filter(Dancer.name == dancer.name).first()
    if existing:
        raise HTTPException(status_code=409, detail="A dancer with this name already exists")
    db_dancer = Dancer(name=dancer.name, experience_level=dancer.experience_level)
    db.add(db_dancer)
    db.commit()
    db.refresh(db_dancer)
    return db_dancer


@router.get("/", response_model=list[DancerResponse])
def list_dancers(db: Session = Depends(get_db)):
    return db.query(Dancer).all()


@router.get("/{dancer_id}", response_model=DancerResponse)
def get_dancer(dancer_id: int, db: Session = Depends(get_db)):
    dancer = db.query(Dancer).filter(Dancer.id == dancer_id).first()
    if not dancer:
        raise HTTPException(status_code=404, detail="Dancer not found")
    return dancer


@router.delete("/{dancer_id}")
def delete_dancer(dancer_id: int, db: Session = Depends(get_db)):
    dancer = db.query(Dancer).filter(Dancer.id == dancer_id).first()
    if not dancer:
        raise HTTPException(status_code=404, detail="Dancer not found")

    # Clean up video files for all performances
    for perf in dancer.performances:
        if perf.video_key:
            video_path = f"/app/uploads/{perf.video_key}"
            if os.path.exists(video_path):
                os.remove(video_path)

    db.delete(dancer)
    db.commit()
    return {"status": "deleted"}

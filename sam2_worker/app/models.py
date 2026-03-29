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
    pose_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class TrackingFrame(Base):
    __tablename__ = "tracking_frames"
    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(
        ForeignKey("performances.id", ondelete="CASCADE")
    )
    dancer_index: Mapped[int] = mapped_column(Integer)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[dict] = mapped_column(JSON)
    mask_iou: Mapped[float] = mapped_column(Float)

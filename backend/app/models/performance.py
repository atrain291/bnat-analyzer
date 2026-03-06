from datetime import datetime

from sqlalchemy import String, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    dancer_id: Mapped[int] = mapped_column(ForeignKey("dancers.id"))
    date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    notes: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    dancer: Mapped["Dancer"] = relationship(back_populates="sessions")  # noqa: F821
    performances: Mapped[list["Performance"]] = relationship(back_populates="session")


class Performance(Base):
    """A single dance video recording within a practice session."""
    __tablename__ = "performances"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id"), nullable=True)
    dancer_id: Mapped[int] = mapped_column(ForeignKey("dancers.id"))

    # Dance metadata
    item_name: Mapped[str | None] = mapped_column(String(300), nullable=True)  # e.g. "Alarippu", "Jatiswaram"
    item_type: Mapped[str | None] = mapped_column(String(100), nullable=True)  # alarippu, jatiswaram, shabdam, varnam, padam, tillana, shlokam
    talam: Mapped[str | None] = mapped_column(String(100), nullable=True)  # e.g. "Adi", "Rupaka", "Misra Chapu"
    ragam: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Video
    video_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    video_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Processing
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, queued, processing, complete, failed
    task_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    pipeline_progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Detection
    detection_frame_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Relationships
    dancer: Mapped["Dancer"] = relationship(back_populates="performances")  # noqa: F821
    session: Mapped["Session | None"] = relationship(back_populates="performances")
    frames: Mapped[list["Frame"]] = relationship(back_populates="performance", cascade="all, delete-orphan")  # noqa: F821
    analysis: Mapped[list["Analysis"]] = relationship(back_populates="performance", cascade="all, delete-orphan")  # noqa: F821
    detected_persons: Mapped[list["DetectedPerson"]] = relationship(back_populates="performance", cascade="all, delete-orphan")
    performance_dancers: Mapped[list["PerformanceDancer"]] = relationship(back_populates="performance", cascade="all, delete-orphan")


class DetectedPerson(Base):
    """A person detected during the detection pass."""
    __tablename__ = "detected_persons"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id", ondelete="CASCADE"))
    track_id: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[dict] = mapped_column(JSON)
    representative_pose: Mapped[dict] = mapped_column(JSON)
    frame_count: Mapped[int] = mapped_column(Integer)
    area: Mapped[float] = mapped_column(Float)

    performance: Mapped["Performance"] = relationship(back_populates="detected_persons")


class PerformanceDancer(Base):
    """A user-confirmed dancer selection for analysis."""
    __tablename__ = "performance_dancers"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id", ondelete="CASCADE"))
    track_id: Mapped[int] = mapped_column(Integer)
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)

    performance: Mapped["Performance"] = relationship(back_populates="performance_dancers")
    frames: Mapped[list["Frame"]] = relationship(back_populates="performance_dancer", cascade="all, delete-orphan")  # noqa: F821
    analysis: Mapped["Analysis | None"] = relationship(back_populates="performance_dancer", uselist=False, cascade="all, delete-orphan")  # noqa: F821

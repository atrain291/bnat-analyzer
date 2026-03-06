from datetime import datetime

from sqlalchemy import String, Integer, DateTime, JSON, ForeignKey, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Performance(Base):
    __tablename__ = "performances"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id"), nullable=True)
    dancer_id: Mapped[int] = mapped_column(ForeignKey("dancers.id"))
    item_name: Mapped[str | None] = mapped_column(String(300), nullable=True)
    item_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    talam: Mapped[str | None] = mapped_column(String(100), nullable=True)
    ragam: Mapped[str | None] = mapped_column(String(100), nullable=True)
    video_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    video_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    task_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    pipeline_progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id"))
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    dancer_pose: Mapped[dict] = mapped_column(JSON)
    left_hand: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    right_hand: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    face: Mapped[list | None] = mapped_column(JSON, nullable=True)


class JointAngleState(Base):
    __tablename__ = "joint_angle_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)

    aramandi_angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    torso_uprightness: Mapped[float | None] = mapped_column(Float, nullable=True)
    arm_extension_left: Mapped[float | None] = mapped_column(Float, nullable=True)
    arm_extension_right: Mapped[float | None] = mapped_column(Float, nullable=True)
    hip_symmetry: Mapped[float | None] = mapped_column(Float, nullable=True)

    all_angles: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class BalanceMetrics(Base):
    __tablename__ = "balance_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)

    center_of_mass_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    center_of_mass_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight_distribution: Mapped[float | None] = mapped_column(Float, nullable=True)
    stability_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id"), unique=True)
    aramandi_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_body_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    symmetry_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rhythm_consistency_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    technique_scores: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    patterns: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    fatigue_markers: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    llm_summary: Mapped[str | None] = mapped_column(String(8000), nullable=True)
    practice_plan: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

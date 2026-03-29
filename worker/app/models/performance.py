from datetime import datetime

from sqlalchemy import String, Integer, DateTime, JSON, ForeignKey, Float, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Dancer(Base):
    __tablename__ = "dancers"
    id: Mapped[int] = mapped_column(primary_key=True)


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)


class Performance(Base):
    __tablename__ = "performances"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id"), nullable=True)
    dancer_id: Mapped[int] = mapped_column(ForeignKey("dancers.id"), index=True)
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
    start_timestamp_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    click_prompts: Mapped[list | None] = mapped_column(JSON, nullable=True)
    beat_timestamps: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tempo_bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    detection_frame_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id"))
    performance_dancer_id: Mapped[int | None] = mapped_column(ForeignKey("performance_dancers.id"), nullable=True, index=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    dancer_pose: Mapped[dict] = mapped_column(JSON)
    left_hand: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    right_hand: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    face: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # WHAM 3D data
    joints_3d: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 24x3 SMPL joints in world/camera coords
    world_position: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {x, y, z} pelvis position
    foot_contact: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {left_heel, left_toe, right_heel, right_toe}


class JointAngleState(Base):
    __tablename__ = "joint_angle_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)

    aramandi_angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    torso_uprightness: Mapped[float | None] = mapped_column(Float, nullable=True)
    arm_extension_left: Mapped[float | None] = mapped_column(Float, nullable=True)
    arm_extension_right: Mapped[float | None] = mapped_column(Float, nullable=True)
    hip_symmetry: Mapped[float | None] = mapped_column(Float, nullable=True)

    # WHAM 3D angles
    knee_angle_3d: Mapped[float | None] = mapped_column(Float, nullable=True)
    torso_angle_3d: Mapped[float | None] = mapped_column(Float, nullable=True)
    hip_abduction_left: Mapped[float | None] = mapped_column(Float, nullable=True)
    hip_abduction_right: Mapped[float | None] = mapped_column(Float, nullable=True)
    torso_twist: Mapped[float | None] = mapped_column(Float, nullable=True)

    all_angles: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class BalanceMetrics(Base):
    __tablename__ = "balance_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)

    center_of_mass_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    center_of_mass_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight_distribution: Mapped[float | None] = mapped_column(Float, nullable=True)
    stability_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # WHAM 3D center of mass
    center_of_mass_3d_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    center_of_mass_3d_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    center_of_mass_3d_z: Mapped[float | None] = mapped_column(Float, nullable=True)


class DetectedPerson(Base):
    __tablename__ = "detected_persons"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[dict] = mapped_column(JSON)
    representative_pose: Mapped[dict] = mapped_column(JSON)
    frame_count: Mapped[int] = mapped_column(Integer)
    area: Mapped[float] = mapped_column(Float)
    appearance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    color_histogram: Mapped[list | None] = mapped_column(JSON, nullable=True)


class TrackingFrame(Base):
    __tablename__ = "tracking_frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id", ondelete="CASCADE"), index=True)
    dancer_index: Mapped[int] = mapped_column(Integer)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[dict] = mapped_column(JSON)
    mask_iou: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        Index("ix_tracking_frames_perf_dancer_ts", "performance_id", "dancer_index", "timestamp_ms"),
    )


class PerformanceDancer(Base):
    __tablename__ = "performance_dancers"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    pose_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id"), index=True)
    performance_dancer_id: Mapped[int | None] = mapped_column(ForeignKey("performance_dancers.id"), nullable=True, index=True)
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

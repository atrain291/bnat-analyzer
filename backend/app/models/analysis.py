from datetime import datetime

from sqlalchemy import String, Integer, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Frame(Base):
    """Per-frame pose data for a performance."""
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id"))
    timestamp_ms: Mapped[int] = mapped_column(Integer)

    # 17-point COCO keypoints: {keypoint_name: {x, y, z, confidence}}
    dancer_pose: Mapped[dict] = mapped_column(JSON)

    performance: Mapped["Performance"] = relationship(back_populates="frames")  # noqa: F821
    joint_angle_state: Mapped["JointAngleState | None"] = relationship(back_populates="frame", uselist=False)
    balance_metrics: Mapped["BalanceMetrics | None"] = relationship(back_populates="frame", uselist=False)
    mudra_state: Mapped["MudraState | None"] = relationship(back_populates="frame", uselist=False)


class JointAngleState(Base):
    """Joint angles computed from pose keypoints for a single frame."""
    __tablename__ = "joint_angle_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)

    # Key angles for Bharatanatyam (degrees)
    aramandi_angle: Mapped[float | None] = mapped_column(Float, nullable=True)  # knee bend angle
    torso_uprightness: Mapped[float | None] = mapped_column(Float, nullable=True)  # torso vertical deviation
    arm_extension_left: Mapped[float | None] = mapped_column(Float, nullable=True)
    arm_extension_right: Mapped[float | None] = mapped_column(Float, nullable=True)
    hip_symmetry: Mapped[float | None] = mapped_column(Float, nullable=True)  # hip alignment deviation

    # Full angle data as JSON for flexibility
    all_angles: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    frame: Mapped["Frame"] = relationship(back_populates="joint_angle_state")


class BalanceMetrics(Base):
    """Balance and weight distribution metrics per frame."""
    __tablename__ = "balance_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)

    center_of_mass_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    center_of_mass_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight_distribution: Mapped[float | None] = mapped_column(Float, nullable=True)  # -1 (left) to 1 (right)
    stability_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0-1 composite

    frame: Mapped["Frame"] = relationship(back_populates="balance_metrics")


class MudraState(Base):
    """Hand gesture (mudra) classification per frame. (Stage 2+)"""
    __tablename__ = "mudra_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)

    left_hand_mudra: Mapped[str | None] = mapped_column(String(100), nullable=True)
    right_hand_mudra: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confidence_left: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_right: Mapped[float | None] = mapped_column(Float, nullable=True)

    frame: Mapped["Frame"] = relationship(back_populates="mudra_state")


class Analysis(Base):
    """LLM-generated coaching analysis for a performance."""
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id"), unique=True)

    # Scores (0-100)
    aramandi_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_body_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    symmetry_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rhythm_consistency_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Detailed analysis
    technique_scores: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    patterns: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    fatigue_markers: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    llm_summary: Mapped[str | None] = mapped_column(String(8000), nullable=True)
    practice_plan: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    performance: Mapped["Performance"] = relationship(back_populates="analysis")  # noqa: F821

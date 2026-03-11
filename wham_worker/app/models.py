"""Minimal SQLAlchemy models — only columns the WHAM worker reads/writes."""

from sqlalchemy import String, Integer, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Performance(Base):
    __tablename__ = "performances"

    id: Mapped[int] = mapped_column(primary_key=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20))


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    performance_id: Mapped[int] = mapped_column(ForeignKey("performances.id"))
    performance_dancer_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    dancer_pose: Mapped[dict] = mapped_column(JSON)

    # Written by WHAM worker
    joints_3d: Mapped[list | None] = mapped_column(JSON, nullable=True)
    world_position: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    foot_contact: Mapped[dict | None] = mapped_column(JSON, nullable=True)

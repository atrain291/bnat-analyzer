from datetime import datetime

from sqlalchemy import String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Dancer(Base):
    __tablename__ = "dancers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    experience_level: Mapped[str | None] = mapped_column(String(50))  # beginner, intermediate, advanced, professional
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    preferences: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    sessions: Mapped[list["Session"]] = relationship(back_populates="dancer", cascade="all, delete-orphan")  # noqa: F821
    performances: Mapped[list["Performance"]] = relationship(back_populates="dancer", cascade="all, delete-orphan")  # noqa: F821

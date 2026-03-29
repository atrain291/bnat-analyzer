import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://bharatanatyam:bharatanatyam@postgres:5432/bharatanatyam_analyzer",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=2, max_overflow=0, pool_recycle=1800)
SessionLocal = sessionmaker(bind=engine)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

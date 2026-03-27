import logging
from app.celery_app import app
from app.db import get_session
from app.models.performance import Performance
from app.pipeline.ingest import ensure_browser_playable

logger = logging.getLogger(__name__)


@app.task(bind=True, name="worker.app.tasks.transcode.run_transcode")
def run_transcode(self, performance_id: int, video_path: str):
    logger.info(f"Transcoding performance {performance_id}: {video_path}")
    try:
        ensure_browser_playable(video_path)
        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "uploaded"
        logger.info(f"Transcode complete for performance {performance_id}")
    except Exception as e:
        logger.error(f"Transcode failed for performance {performance_id}: {e}")
        with get_session() as session:
            perf = session.query(Performance).filter(Performance.id == performance_id).first()
            if perf:
                perf.status = "failed"
                perf.error = f"Video transcode failed: {str(e)[:500]}"

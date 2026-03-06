from datetime import datetime

from app.schemas.dancer import DancerCreate, DancerResponse
from app.schemas.performance import (
    PerformanceStatusResponse,
    FrameResponse,
    AnalysisResponse,
    UploadResponse,
)


def test_dancer_create():
    d = DancerCreate(name="Meera", experience_level="beginner")
    assert d.name == "Meera"
    assert d.experience_level == "beginner"


def test_dancer_create_no_level():
    d = DancerCreate(name="Priya")
    assert d.experience_level is None


def test_dancer_response():
    d = DancerResponse(id=1, name="Meera", experience_level="advanced", created_at=datetime.now())
    assert d.id == 1


def test_performance_status_response():
    r = PerformanceStatusResponse(id=1, status="processing", pipeline_progress={"stage": "pose"}, error=None)
    assert r.status == "processing"


def test_frame_response():
    f = FrameResponse(id=1, timestamp_ms=500, dancer_pose={"nose": {"x": 0.5, "y": 0.3}})
    assert f.timestamp_ms == 500


def test_analysis_response():
    a = AnalysisResponse(
        id=1,
        aramandi_score=80.0,
        upper_body_score=None,
        symmetry_score=None,
        rhythm_consistency_score=None,
        overall_score=None,
        technique_scores=None,
        llm_summary="Good.",
        practice_plan=None,
        created_at=datetime.now(),
    )
    assert a.aramandi_score == 80.0


def test_upload_response():
    r = UploadResponse(performance_id=1, task_id="abc-123", status="queued")
    assert r.task_id == "abc-123"

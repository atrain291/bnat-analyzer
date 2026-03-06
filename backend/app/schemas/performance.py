from datetime import datetime
from pydantic import BaseModel, ConfigDict


class PerformanceStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    pipeline_progress: dict | None
    error: str | None


class FrameResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp_ms: int
    dancer_pose: dict


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    aramandi_score: float | None
    upper_body_score: float | None
    symmetry_score: float | None
    rhythm_consistency_score: float | None
    overall_score: float | None
    technique_scores: dict | None
    llm_summary: str | None
    practice_plan: dict | None
    created_at: datetime


class PerformanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dancer_id: int
    item_name: str | None
    item_type: str | None
    talam: str | None
    ragam: str | None
    video_url: str | None
    duration_ms: int | None
    status: str
    pipeline_progress: dict | None
    error: str | None
    created_at: datetime
    frames: list[FrameResponse] = []
    analysis: list[AnalysisResponse] = []


class UploadResponse(BaseModel):
    performance_id: int
    task_id: str
    status: str

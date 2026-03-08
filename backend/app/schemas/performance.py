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
    performance_dancer_id: int | None = None


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    performance_dancer_id: int | None = None
    aramandi_score: float | None
    upper_body_score: float | None
    symmetry_score: float | None
    rhythm_consistency_score: float | None
    overall_score: float | None
    technique_scores: dict | None
    llm_summary: str | None
    practice_plan: dict | None
    created_at: datetime


class DetectedPersonResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    track_id: int
    bbox: dict
    representative_pose: dict
    frame_count: int
    area: float


class PerformanceDancerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    track_id: int
    label: str | None


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
    detection_frame_url: str | None = None
    created_at: datetime
    analysis: list[AnalysisResponse] = []
    detected_persons: list[DetectedPersonResponse] = []
    performance_dancers: list[PerformanceDancerResponse] = []


class PerformanceListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dancer_id: int
    item_name: str | None
    item_type: str | None
    status: str
    overall_score: float | None = None
    duration_ms: int | None
    created_at: datetime


class TimelineFrameResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp_ms: int
    performance_dancer_id: int | None = None
    aramandi_angle: float | None = None
    torso_uprightness: float | None = None
    arm_extension_left: float | None = None
    arm_extension_right: float | None = None
    hip_symmetry: float | None = None
    stability_score: float | None = None


class DancerSelectionItem(BaseModel):
    track_id: int
    label: str | None = None


class DancerSelectionRequest(BaseModel):
    selections: list[DancerSelectionItem]


class UploadResponse(BaseModel):
    performance_id: int
    task_id: str
    status: str

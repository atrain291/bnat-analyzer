from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DancerCreate(BaseModel):
    name: str
    experience_level: str | None = None


class DancerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    experience_level: str | None
    created_at: datetime

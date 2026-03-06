from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://bharatanatyam:bharatanatyam@postgres:5432/bharatanatyam_analyzer"
    redis_url: str = "redis://redis:6379/0"
    secret_key: str = "change-me-in-production"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    anthropic_api_key: str = ""
    s3_bucket: str = ""
    s3_region: str = ""
    s3_endpoint_url: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()

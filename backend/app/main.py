from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.routes import dancers, performances, upload

app = FastAPI(title="Bharatanatyam Dance Analyzer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dancers.router)
app.include_router(performances.router)
app.include_router(upload.router)

app.mount("/uploads", StaticFiles(directory="/app/uploads"), name="uploads")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "bharatanatyam-analyzer"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import upload, analysis, training, tuning, settings, artifacts

app = FastAPI(title="ML-TEST", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/data", tags=["Data Upload"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["EDA"])
app.include_router(training.router, prefix="/api/train", tags=["Training"])
app.include_router(tuning.router, prefix="/api/tuning", tags=["LLM Tuning"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(artifacts.router, prefix="/api/artifacts", tags=["Artifacts"])


@app.get("/")
async def root():
    return {"status": "ok", "message": "ML-TEST API v2 running"}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}

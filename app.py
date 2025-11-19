# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict
import uvicorn

from orchestrator import run_job_sync
from storage import read_job

app = FastAPI(title="EVA - Backend (Gemini-powered)", version="0.2.0")

app.add_middleware(
    CORSMiddleware(
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
)


class TargetSpec(BaseModel):
    format: str = "3_bullets"
    max_words: int = 60


class ProblemRequest(BaseModel):
    user_id: str = "team_hexad"
    problem_text: str
    domain: str = "lecture"
    target: Optional[TargetSpec] = None
    n_agents: int = 3
    style_hint: Optional[str] = None
    seed: Optional[int] = 42


@app.get("/health")
def health():
    return {"status": "ok", "service": "eva-backend-gemini"}


@app.post("/request")
def create_request(req: ProblemRequest) -> Dict[str, Any]:
    if not req.problem_text.strip():
        raise HTTPException(status_code=400, detail="problem_text is required")

    job = run_job_sync(req.dict())
    return job


@app.get("/job/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = read_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

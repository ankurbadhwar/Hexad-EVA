import os
import json
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "storage", "jobs")
BPS_DIR = os.path.join(BASE_DIR, "storage", "blueprints")

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(BPS_DIR, exist_ok=True)


def _safe_write(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def write_job(job_id: str, job: Dict[str, Any]) -> None:
    path = os.path.join(JOBS_DIR, f"{job_id}.json")
    _safe_write(path, job)


def read_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(JOBS_DIR, f"{job_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_blueprint(bp_id: str, bp: Dict[str, Any]) -> None:
    path = os.path.join(BPS_DIR, f"{bp_id}.json")
    _safe_write(path, bp)

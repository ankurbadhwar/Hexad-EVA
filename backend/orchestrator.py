# backend/orchestrator.py
"""
Orchestrator for EVO-EVA.

Responsibilities:
- Coordinate a full synchronous run for a single user request:
    1. Create job id and persist job metadata.
    2. Call Architect to generate N blueprints (fallback to canned if needed).
    3. Persist blueprints.
    4. For each blueprint: build agent, evaluate on testcases, record outputs & score.
    5. Select best blueprint from generation 1.
    6. Mutate best blueprint (one mutation), build & evaluate mutated blueprint.
    7. Persist full results and return final job dictionary.

Design goals:
- Durable: write job state to storage after each major step so you can recover / demo logs.
- Defensible: catch exceptions at each step, persist error traces as part of job record.
- Testable: small helper functions, limited side effects, typed signatures.
"""

from __future__ import annotations
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

# Local imports (expected to exist in project)
from backend import config
from backend import storage
from backend import architect
from backend import builder
from backend import evaluator
from backend import evolution
from backend.utils import get_logger, LLMError

logger = get_logger(__name__)

# Constants
DEFAULT_CANDIDATES = 3


# --- Helper functions -------------------------------------------------------
def _now_ts() -> float:
    return time.time()


def _init_job_record(request_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create initial job record skeleton."""
    return {
        "job_id": str(uuid.uuid4()),
        "created_at": _now_ts(),
        "status": "started",
        "request": request_payload,
        "steps": [],
        "gen1_results": [],
        "mutated_result": None,
        "raw_architect_text": None,
        "error": None,
    }


def _persist_step(job_id: str, job_obj: Dict[str, Any], step_name: str) -> None:
    """Persist job and append step marker (atomic write via storage.write_job)."""
    job_obj.setdefault("steps", []).append({"step": step_name, "ts": _now_ts()})
    storage.write_job(job_id, job_obj)
    logger.info("persisted job step", extra={"job_id": job_id, "step": step_name})


def _record_exception(job_obj: Dict[str, Any], exc: Exception, context: str) -> None:
    """Attach exception details to job record for debugging / demo transparency."""
    tb = traceback.format_exc()
    job_obj["error"] = {
        "context": context,
        "message": str(exc),
        "traceback": tb,
        "ts": _now_ts(),
    }


# --- Orchestration core ----------------------------------------------------
def run_job_sync(request_payload: Dict[str, Any], n_candidates: int = DEFAULT_CANDIDATES) -> Dict[str, Any]:
    """
    Full synchronous orchestration run.

    Args:
        request_payload: dict with user request fields (user_id, problem_text, domain, target, etc.)
        n_candidates: number of candidate blueprints Architect should produce

    Returns:
        job_obj: final persisted job dictionary with results and metadata
    """
    job = _init_job_record(request_payload)
    job_id = job["job_id"]
    logger.info("starting job", extra={"job_id": job_id})

    # Ensure storage directories exist (storage layer should handle this, but double-check)
    storage.ensure_storage_exists()

    # Persist initial job
    _persist_step(job_id, job, "job_created")

    # --- 1) Architect: generate blueprints ---------------------------------
    try:
        logger.info("architect: generating blueprints", extra={"job_id": job_id})
        job["steps"].append({"note": "architect_start", "ts": _now_ts()})
        # architect.generate_blueprints should return a list[dict] or raise
        blueprints, raw_text = architect.generate_blueprints_with_debug(request_payload, n_candidates)
        job["raw_architect_text"] = raw_text
        # Save blueprints to storage
        saved_bp_ids = []
        for bp in blueprints:
            storage.save_blueprint(bp)
            saved_bp_ids.append(bp.get("id"))
        job["blueprint_ids"] = saved_bp_ids
        _persist_step(job_id, job, "architect_done")
        logger.info("architect: done", extra={"job_id": job_id, "blueprint_count": len(blueprints)})
    except Exception as exc:
        # Architect failed — fallback to canned blueprints (architect module should provide fallback, but be safe)
        logger.exception("architect failed, using canned fallback", extra={"job_id": job_id})
        _record_exception(job, exc, "architect_failure")
        # Try to load canned blueprints from templates via storage helper or architect helper
        try:
            blueprints = architect.load_canned_blueprints()
            for bp in blueprints:
                storage.save_blueprint(bp)
            job["blueprint_ids"] = [bp.get("id") for bp in blueprints]
            job["raw_architect_text"] = "<architect_error_fallback_used>"
            _persist_step(job_id, job, "architect_fallback_used")
        except Exception as inner_exc:
            logger.exception("canned blueprint fallback also failed", extra={"job_id": job_id})
            _record_exception(job, inner_exc, "canned_blueprints_failure")
            job["status"] = "error"
            storage.write_job(job_id, job)
            return job  # unrecoverable

    # --- Load evaluation testcases -----------------------------------------
    try:
        testcases = evaluator.load_testcases()
    except Exception as exc:
        logger.exception("failed to load testcases", extra={"job_id": job_id})
        _record_exception(job, exc, "load_testcases_failure")
        job["status"] = "error"
        _persist_step(job_id, job, "testcases_load_failed")
        return job

    # --- 2) Build & Evaluate generation 1 candidates -----------------------
    gen1_results = []
    for bp in blueprints:
        bp_id = bp.get("id", "<no-id>")
        try:
            logger.info("building agent from blueprint", extra={"job_id": job_id, "blueprint_id": bp_id})
            agent_obj = builder.build_agent_from_blueprint(bp)

            logger.info("evaluating agent", extra={"job_id": job_id, "blueprint_id": bp_id})
            ev = evaluator.evaluate_agent(agent_obj, testcases)
            # expected ev: {"avg_score": float, "outputs": [{...}], "meta": {...}}
            entry = {
                "blueprint_id": bp_id,
                "blueprint": bp,
                "avg_score": ev.get("avg_score"),
                "outputs": ev.get("outputs"),
                "eval_meta": ev.get("meta", {}),
            }
            gen1_results.append(entry)
            job.setdefault("gen1_results", []).append(entry)
            _persist_step(job_id, job, f"evaluated_{bp_id}")
            logger.info("evaluated candidate", extra={"job_id": job_id, "blueprint_id": bp_id, "score": entry["avg_score"]})
        except Exception as exc:
            logger.exception("error building or evaluating candidate", extra={"job_id": job_id, "blueprint_id": bp_id})
            _record_exception(job, exc, f"evaluate_failure_{bp_id}")
            # continue to next blueprint without failing entire job
            job.setdefault("gen1_results", []).append({
                "blueprint_id": bp_id,
                "blueprint": bp,
                "avg_score": 0.0,
                "outputs": [],
                "error": str(exc),
            })
            _persist_step(job_id, job, f"evaluate_error_{bp_id}")

    # If no gen1 results (shouldn't happen), fail gracefully
    if not gen1_results:
        job["status"] = "error"
        job["error"] = {"message": "no gen1 candidates evaluated"}
        storage.write_job(job_id, job)
        return job

    # --- 3) Select best candidate -----------------------------------------
    try:
        # deterministic selection: max avg_score (tie-breaker by first occurrence)
        best_entry = max(gen1_results, key=lambda e: (e.get("avg_score") or 0.0))
        job["best_gen1_blueprint_id"] = best_entry.get("blueprint_id")
        job["best_gen1_score"] = best_entry.get("avg_score")
        _persist_step(job_id, job, "selected_best_gen1")
        logger.info("selected best gen1", extra={"job_id": job_id, "best_bp": job["best_gen1_blueprint_id"], "score": job["best_gen1_score"]})
    except Exception as exc:
        logger.exception("selection failed", extra={"job_id": job_id})
        _record_exception(job, exc, "selection_failure")
        job["status"] = "error"
        storage.write_job(job_id, job)
        return job

    # --- 4) Mutate best blueprint and evaluate mutated ---------------------
    try:
        best_bp = best_entry.get("blueprint")
        logger.info("mutating best blueprint", extra={"job_id": job_id, "blueprint_id": best_bp.get("id")})
        mutated_bp = evolution.mutate_blueprint(best_bp)
        storage.save_blueprint(mutated_bp)
        _persist_step(job_id, job, "mutated_blueprint_saved")
    except Exception as exc:
        logger.exception("mutation failed", extra={"job_id": job_id})
        _record_exception(job, exc, "mutation_failure")
        # Safe fallback: set mutated_result to None and return gen1 results
        job["mutated_result"] = {"error": "mutation_failed", "fallback_best": best_entry}
        job["status"] = "done"
        storage.write_job(job_id, job)
        return job

    # Evaluate mutated blueprint
    try:
        mutated_agent = builder.build_agent_from_blueprint(mutated_bp)
        mutated_eval = evaluator.evaluate_agent(mutated_agent, testcases)
        mutated_result = {
            "blueprint_id": mutated_bp.get("id"),
            "blueprint": mutated_bp,
            "avg_score": mutated_eval.get("avg_score"),
            "outputs": mutated_eval.get("outputs"),
            "eval_meta": mutated_eval.get("meta", {}),
        }
        job["mutated_result"] = mutated_result
        _persist_step(job_id, job, "mutated_evaluation_done")
        logger.info("mutated evaluation done", extra={"job_id": job_id, "mutated_bp": mutated_bp.get("id"), "score": mutated_result["avg_score"]})
    except Exception as exc:
        logger.exception("mutated evaluation failed", extra={"job_id": job_id})
        _record_exception(job, exc, "mutated_evaluation_failure")
        job["mutated_result"] = {"error": "mutated_evaluation_failed", "fallback_best": best_entry}
        _persist_step(job_id, job, "mutated_evaluation_failed")
        job["status"] = "done"
        storage.write_job(job_id, job)
        return job

    # --- 5) Finalize job --------------------------------------------------
    job["status"] = "done"
    job["finished_at"] = _now_ts()
    # Convenience fields: chosen final agent (mutated if better, else gen1 best)
    try:
        mutated_score = job["mutated_result"].get("avg_score") if job.get("mutated_result") else None
        gen1_best_score = job.get("best_gen1_score", 0.0)
        if mutated_score is not None and mutated_score >= gen1_best_score:
            job["final_choice"] = {
                "blueprint_id": job["mutated_result"]["blueprint_id"],
                "method": "mutated",
                "score": mutated_score,
            }
        else:
            job["final_choice"] = {
                "blueprint_id": job["best_gen1_blueprint_id"],
                "method": "gen1_best",
                "score": gen1_best_score,
            }
    except Exception:
        # non-fatal - still persist job
        logger.exception("final choice calculation failed", extra={"job_id": job_id})

    storage.write_job(job_id, job)
    logger.info("job completed", extra={"job_id": job_id, "final_choice": job.get("final_choice")})
    return job

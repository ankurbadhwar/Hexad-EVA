# orchestrator.py
import uuid
from typing import Dict, Any, List, Tuple, Optional

from architect import generate_blueprints
from builder import build_agent_from_blueprint
from evaluator import evaluate_agent
from evolution import mutate_blueprint
from storage import write_job, write_blueprint
from data_loader import load_testcases
from codegen import blueprint_to_python_code


def _evaluate_blueprints(blueprints: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    testcases = load_testcases()
    gen1_results: List[Dict[str, Any]] = []

    for bp in blueprints:
        agent = build_agent_from_blueprint(bp)
        eval_result = evaluate_agent(agent, testcases)
        gen1_results.append({
            "blueprint_id": bp["id"],
            "blueprint_name": bp.get("name"),
            "avg_score": eval_result["avg_score"],
            "metrics": eval_result,
        })

    best = max(gen1_results, key=lambda r: r["avg_score"]) if gen1_results else None
    return gen1_results, best


def run_job_sync(problem_spec: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())

    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "started",
        "problem_spec": problem_spec,
        "gen1_results": [],
        "mutated_result": None,
        "final_agent": None,
        "agent_code": None,
    }
    write_job(job_id, job)

    n_agents = int(problem_spec.get("n_agents", 3))
    blueprints = generate_blueprints(problem_spec, n=n_agents)
    for bp in blueprints:
        write_blueprint(bp["id"], bp)

    gen1_results, best_gen1 = _evaluate_blueprints(blueprints)

    mutated_result = None
    mutated_bp = None

    if best_gen1:
        best_bp = next(bp for bp in blueprints if bp["id"] == best_gen1["blueprint_id"])
        mutated_bp = mutate_blueprint(best_bp)
        write_blueprint(mutated_bp["id"], mutated_bp)

        testcases = load_testcases()
        mutated_agent = build_agent_from_blueprint(mutated_bp)
        mutated_eval = evaluate_agent(mutated_agent, testcases)

        mutated_result = {
            "blueprint_id": mutated_bp["id"],
            "blueprint_name": mutated_bp.get("name"),
            "avg_score": mutated_eval["avg_score"],
            "metrics": mutated_eval,
        }

    final_source = None
    final_bp = None
    final_avg_score = None

    if best_gen1 and mutated_result:
        if mutated_result["avg_score"] > best_gen1["avg_score"]:
            final_source = "mutated"
            final_bp = mutated_bp
            final_avg_score = mutated_result["avg_score"]
        else:
            final_source = "gen1"
            final_bp = next(bp for bp in blueprints if bp["id"] == best_gen1["blueprint_id"])
            final_avg_score = best_gen1["avg_score"]
    elif best_gen1:
        final_source = "gen1"
        final_bp = next(bp for bp in blueprints if bp["id"] == best_gen1["blueprint_id"])
        final_avg_score = best_gen1["avg_score"]
    elif mutated_result:
        final_source = "mutated"
        final_bp = mutated_bp
        final_avg_score = mutated_result["avg_score"]

    final_agent_meta = None
    agent_code = None

    if final_bp is not None:
        final_agent_meta = {
            "source": final_source,
            "blueprint_id": final_bp["id"],
            "blueprint_name": final_bp.get("name"),
            "avg_score": final_avg_score,
            "generation": final_bp.get("metadata", {}).get("generation"),
        }
        agent_code = blueprint_to_python_code(final_bp)

    job["status"] = "done"
    job["gen1_results"] = gen1_results
    job["mutated_result"] = mutated_result
    job["final_agent"] = final_agent_meta
    job["agent_code"] = agent_code

    write_job(job_id, job)
    return job

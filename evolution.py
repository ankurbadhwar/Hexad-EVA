# evolution.py
from typing import Dict, Any
import copy


def mutate_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple, visible mutation:
    - Enforce concision
    - Enforce exactly 3 bullets
    - Decrease temperature a bit
    """
    mutated = copy.deepcopy(bp)
    pt1 = mutated["prompt_templates"]["pt1"]

    if "Be concise." not in pt1:
        pt1 = "Be concise. " + pt1

    if "exactly 3 bullet" not in pt1.lower():
        pt1 += "\nOutput must be exactly 3 bullet points."

    mutated["prompt_templates"]["pt1"] = pt1

    temp = mutated.get("temperature", 0.3)
    mutated["temperature"] = max(0.05, temp - 0.1)

    meta = mutated.get("metadata", {})
    meta["generation"] = meta.get("generation", 1) + 1
    meta["mutated_from"] = bp["id"]
    mutated["metadata"] = meta

    mutated["id"] = f"{bp['id']}_mut"
    mutated["name"] = bp.get("name", "Agent") + " (evolved)"

    return mutated

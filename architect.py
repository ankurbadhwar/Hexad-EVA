# builder.py
from typing import Dict, Any

from utils import call_llm, postprocess_text


def build_agent_from_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn a blueprint into a runnable agent:
    agent["run"](input_text: str) -> str
    """

    def run(text: str) -> str:
        prompt_template = bp["prompt_templates"]["pt1"]
        prompt = prompt_template.replace("{text}", text)

        raw_output = call_llm(
            prompt=prompt,
            temperature=bp.get("temperature", 0.2),
            max_tokens=bp.get("max_tokens", 300),
        )

        processed = postprocess_text(
            raw_output,
            max_words=bp.get("postprocess", {}).get("max_words"),
            strip_lines=bp.get("postprocess", {}).get("strip_lines", True),
        )
        return processed

    agent: Dict[str, Any] = {
        "id": bp["id"],
        "name": bp.get("name", "UnnamedAgent"),
        "blueprint": bp,
        "run": run,  # type: ignore
    }
    return agent

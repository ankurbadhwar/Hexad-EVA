# backend/architect.py
"""
Architect module for EVO-EVA.

Primary responsibilities:
- Generate N candidate blueprints from a ProblemSpec using Gemini (LLM).
- Return both the parsed blueprints and the raw LLM text for debugging.
- Validate output using the Pydantic Blueprint model, and fall back to canned blueprints on any error.

Public functions:
- generate_blueprints_with_debug(problem_spec: dict, n: int) -> (List[dict], str)
- load_canned_blueprints() -> List[dict]

Notes:
- This module intentionally returns plain dicts (not Pydantic objects) so storage writes are JSON-serializable.
- Keep the architect prompt in templates/architect_prompt.txt; canned fallbacks live in templates/canned_blueprints.json.
"""

from __future__ import annotations
import json
import os
import logging
import traceback
from typing import Any, Dict, List, Tuple

from backend import config
from backend import storage
from backend.utils import call_gemini, get_logger, sanitize_text_for_json
from backend.schemas import Blueprint  # Pydantic model for validation

logger = get_logger(__name__)

# Template paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
ARCHITECT_PROMPT_PATH = os.path.join(TEMPLATES_DIR, "architect_prompt.txt")
CANNED_BLUEPRINTS_PATH = os.path.join(TEMPLATES_DIR, "canned_blueprints.json")


# --------------------- Utilities --------------------------------------------
def _read_prompt_template() -> str:
    """Load architect prompt template from templates/architect_prompt.txt."""
    if not os.path.exists(ARCHITECT_PROMPT_PATH):
        raise FileNotFoundError(f"Architect prompt template missing: {ARCHITECT_PROMPT_PATH}")
    with open(ARCHITECT_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _extract_first_json_array(text: str) -> str:
    """
    Extract the first JSON array substring from free text.
    Returns substring (including square brackets) or raises ValueError if not found.

    This function uses a simple bracket stack walk to find a balanced top-level array.
    """
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array '[' found in text")
    stack = []
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            stack.append(i)
        elif ch == "]":
            stack.pop()
            if not stack:
                return text[start:i + 1]
    raise ValueError("No balanced JSON array found in text")


def _parse_blueprints_json(json_text: str) -> List[Dict[str, Any]]:
    """
    Parse JSON array text into list of dicts and validate each blueprint using Pydantic model.
    Returns validated list of plain dicts.
    """
    raw = json.loads(json_text)
    if not isinstance(raw, list):
        raise ValueError("Expected a JSON array of blueprints")
    validated = []
    errors = []
    for idx, item in enumerate(raw):
        try:
            bp = Blueprint.parse_obj(item)
            # convert back to dict (ensures defaults and field normalization)
            validated.append(bp.dict())
        except Exception as e:
            errors.append({"index": idx, "error": str(e), "item": item})
    if errors:
        raise ValueError(f"Blueprint validation failed: {errors}")
    return validated


# --------------------- Public API -------------------------------------------
def generate_blueprints_with_debug(problem_spec: Dict[str, Any], n: int = 3) -> Tuple[List[Dict[str, Any]], str]:
    """
    Ask Gemini to generate `n` blueprints for the given problem_spec.

    Returns:
        (blueprints_list, raw_architect_text)

    Where:
      - blueprints_list: list of validated blueprint dicts (Pydantic-validated)
      - raw_architect_text: the raw text returned by the LLM (for debugging / storage)

    Fallback:
      - If LLM call fails or JSON extraction/validation fails, this function raises the exception to the caller.
        The caller (orchestrator) is expected to catch and perform fallback using load_canned_blueprints().
    """
    logger.info("architect.generate_blueprints_with_debug called", extra={"n": n})
    prompt_template = _read_prompt_template()
    # Fill prompt template safely
    try:
        problem_json = json.dumps(problem_spec, indent=None, ensure_ascii=False)
    except Exception:
        problem_json = str(problem_spec)
    prompt = prompt_template.format(problem_spec_json=problem_json, n=n)

    # Call LLM (Gemini) via utils wrapper
    raw_text = None
    try:
        raw_text = call_gemini(
            prompt=prompt,
            model=config.LLM.model,
            max_tokens=config.LLM.max_tokens,
            temperature=config.LLM.temperature if hasattr(config.LLM, "temperature") else 0.3,
            timeout=config.LLM.timeout_s,
            retries=config.LLM.retries,
        )
        # sanitize small control characters that break JSON parsing
        raw_text = sanitize_text_for_json(raw_text)
        logger.debug("architect raw_text length", extra={"len": len(raw_text)})
    except Exception as exc:
        logger.exception("LLM call in architect failed", extra={"error": str(exc)})
        # bubble up the exception to be handled by caller (orchestrator) which will use canned fallback
        raise

    # Attempt to extract JSON array from raw_text
    try:
        json_array_text = _extract_first_json_array(raw_text)
    except Exception as exc:
        logger.exception("Failed to extract JSON array from architect raw_text", extra={"error": str(exc)})
        raise ValueError(f"JSON extraction error: {exc}") from exc

    # Attempt to parse and validate blueprints
    try:
        blueprints = _parse_blueprints_json(json_array_text)
        # Ensure each blueprint has an id; Pydantic default should cover it but double-check
        for bp in blueprints:
            if "id" not in bp or not bp["id"]:
                bp["id"] = str(storage.make_new_blueprint_id()) if hasattr(storage, "make_new_blueprint_id") else str(uuid.uuid4())
        return blueprints, raw_text
    except Exception as exc:
        logger.exception("Parsing/validation of blueprints failed", extra={"error": str(exc)})
        raise


def load_canned_blueprints() -> List[Dict[str, Any]]:
    """
    Load canned blueprints from templates/canned_blueprints.json.
    Returns a list of blueprint dicts validated by the Pydantic model.

    This function is safe to call as a fallback when the LLM cannot produce valid output.
    """
    logger.info("Loading canned blueprints as fallback", extra={})
    if not os.path.exists(CANNED_BLUEPRINTS_PATH):
        raise FileNotFoundError(f"Canned blueprints not found at {CANNED_BLUEPRINTS_PATH}")
    with open(CANNED_BLUEPRINTS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Validate the canned entries too (ensures they follow current schema)
    validated = []
    errors = []
    for idx, item in enumerate(raw):
        try:
            bp = Blueprint.parse_obj(item)
            validated.append(bp.dict())
        except Exception as e:
            errors.append({"index": idx, "error": str(e), "item": item})
    if errors:
        logger.warning("Some canned blueprints failed validation; discarding invalid ones", extra={"errors": errors})
    if not validated:
        raise ValueError("No valid canned blueprints available after validation")
    # Persist loaded canned blueprints to storage for debugging / simplicity (optional)
    for bp in validated:
        try:
            storage.save_blueprint(bp)
        except Exception:
            logger.debug("saving canned blueprint failed but continuing", exc_info=True)
    return validated

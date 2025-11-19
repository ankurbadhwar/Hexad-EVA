# app.py
import os
import re
import time
import json
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# OpenAI SDK (optional; fallback to mock if not configured)
try:
    from openai import OpenAI
    from openai import RateLimitError, OpenAIError
except Exception:
    OpenAI = None
    RateLimitError = Exception
    OpenAIError = Exception

# ----------------------
# Config & logging
# ----------------------
load_dotenv()
logger = logging.getLogger("hexad.eva")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Running in mock mode (local testing).")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY and OpenAI is not None else None

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI(title="Hexad-EVA (code-only /run)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Models
# ----------------------
class UserProblem(BaseModel):
    problem: str

# ----------------------
# Helpers
# ----------------------
def extract_first_json(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    arr = re.search(r"(\[\s*\{.*?\}\s*\])", text, flags=re.DOTALL)
    if arr:
        return arr.group(1)
    obj = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if obj:
        return obj.group(1)
    return None

def _strip_code_fence(text: str) -> str:
    """
    Strip triple-backtick fences and return paste-ready code (no fences).
    If there are multiple fenced blocks, join them with two newlines.
    """
    if not isinstance(text, str):
        return ""
    s = text.strip()
    # Find all fenced code blocks
    fences = re.findall(r"```[a-zA-Z0-9]*\n(.*?)\n```", s, flags=re.DOTALL)
    if fences:
        return "\n\n".join(f.strip() for f in fences)
    # If no fenced blocks, try to remove loose leading/trailing ``` or single backticks
    s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    if s.startswith("`") and s.endswith("`"):
        s = s[1:-1].strip()
    return s.strip()

def _try_parse_json_from_text(text: str) -> Optional[Any]:
    if not text or not isinstance(text, str):
        return None
    try:
        candidate = extract_first_json(text)
        if candidate:
            return json.loads(candidate)
    except Exception:
        pass
    return None

def _normalize_output_format(ofmt):
    if isinstance(ofmt, list):
        try:
            return "; ".join(str(x) for x in ofmt)
        except Exception:
            return str(ofmt)
    return str(ofmt) if ofmt is not None else ""

def _looks_like_code_format(ofmt: Optional[str]) -> bool:
    if not ofmt:
        return False
    s = ofmt.lower()
    return any(k in s for k in ["code", "script", "python", "bash", "sh", "node", "js", "javascript", "typescript", "java", "runnable"])

# ----------------------
# LLM wrapper (mock-friendly)
# ----------------------
def llm(prompt: str, model: str = "gpt-4o-mini", max_retries: int = 1, temperature: float = 0.7) -> str:
    # Mock behavior when no client: return runnable python code if the prompt asks for code
    if client is None:
        logger.info("LLM mock: generating deterministic mock response.")
        low = prompt.lower()
        if "return only" in low and ("code" in low or "script" in low or "python" in low):
            return (
                "```python\n"
                "# run: python mock_agent.py\n"
                "def agent_main():\n"
                "    print('MockAgent: this is runnable mock python code')\n\n"
                "if __name__ == '__main__':\n"
                "    agent_main()\n"
                "```"
            )
        # default: small mock blueprint JSON
        return json.dumps([
            {
                "name": "MockAgent",
                "objective": "Mock: perform the requested task",
                "reasoning_style": "mock",
                "rules": ["Be concise", "Return runnable code when asked"],
                "output_format": "python script"
            }
        ])
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = response.choices[0].message.content
            return text or ""
        except RateLimitError as e:
            last_err = e
            logger.warning("RateLimitError: %s", e)
            time.sleep(0.3)
        except OpenAIError as e:
            last_err = e
            logger.error("OpenAIError: %s", e)
            time.sleep(0.3)
        except Exception as e:
            last_err = e
            logger.exception("Unexpected error calling LLM")
            time.sleep(0.2)
    logger.error("LLM failed after retries: %s", last_err)
    return ""

# ----------------------
# Pipeline functions
# ----------------------
def generate_blueprints(problem: str, retries: int = 2) -> List[Dict[str, Any]]:
    prompt = f"""
You are EVA - an AI architect who generates BLUEPRINTS for task-specific agents.

Given the user task: "{problem}"

Return EXACTLY a JSON array of 3 objects (no extra commentary). Each object must contain these fields:
- name (short string)
- objective (one-sentence)
- reasoning_style (short phrase)
- rules (array of short rules)
- output_format (explicit; e.g. "python script", "bash script", "json", "plain text")

If the agent should return runnable code, set output_format to a language + 'script' (for example 'python script').
"""
    last_raw = ""
    for attempt in range(retries + 1):
        last_raw = llm(prompt, model="gpt-4o-mini", max_retries=1, temperature=0.5)
        logger.info("generate_blueprints raw (trunc): %s", str(last_raw)[:800])
        candidate = extract_first_json(last_raw)
        if candidate:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and len(parsed) >= 1:
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except Exception as e:
                logger.warning("Failed to parse generated blueprints JSON: %s", e)
        time.sleep(0.3)
    # fallback
    logger.warning("Using fallback blueprint.")
    return [
        {"name": "FallbackAgent", "objective": problem, "reasoning_style": "direct", "rules": ["Be concise"], "output_format": "python script"}
    ]

def run_agent(blueprint: Dict[str, Any], task_input: str) -> str:
    bp_text = json.dumps(blueprint, ensure_ascii=False, indent=2)
    ofmt = (blueprint.get("output_format") or "").strip()
    wants_code = _looks_like_code_format(ofmt)

    prompt = f"""
You are an AI agent with the following blueprint specification:
{bp_text}

User Task Input:
{task_input}

Follow the blueprint's rules and output in the blueprint's output_format.
"""

    if wants_code:
        # ask LLM to return only runnable code inside a single fenced block (language tag)
        lang_hint = ""
        if "python" in ofmt.lower():
            lang_hint = "python"
        elif "bash" in ofmt.lower() or "sh" in ofmt.lower():
            lang_hint = "bash"
        elif "js" in ofmt.lower() or "javascript" in ofmt.lower() or "node" in ofmt.lower():
            lang_hint = "javascript"

        prompt += f"""

IMPORTANT:
- Return ONLY valid, runnable source code inside a single fenced code block (triple backticks) with the correct language tag (e.g. ```{lang_hint}).
- Do NOT include explanatory text or JSON wrappers outside the fenced code block.
- The code should be paste-ready. Include a single-line comment at the top showing how to run it (example: '# run: python script.py').
"""
    else:
        prompt += """
Return the result exactly in the requested output_format. No additional commentary.
"""

    out = llm(prompt, model="gpt-4o-mini", max_retries=1, temperature=0.6)
    logger.info("run_agent (%s) raw (trunc): %s", blueprint.get("name"), str(out)[:600])
    return out or ""

def score_response(problem: str, output: str) -> float:
    prompt = f"""
You are a strict numeric evaluator.

Task: "{problem}"
Agent Output:
""" + output + """

Score this output from 1.0 to 10.0 (float) based on correctness, usefulness, clarity, and alignment.
Return ONLY a single JSON object like: {{"score": 8.2}}
"""
    raw = llm(prompt, model="gpt-4o-mini", max_retries=1, temperature=0.0)
    logger.info("score_response raw (trunc): %s", str(raw)[:400])
    if not raw:
        return 0.0
    try:
        js = extract_first_json(raw)
        if js:
            parsed = json.loads(js)
            if isinstance(parsed, dict) and "score" in parsed:
                return float(parsed["score"]) if parsed["score"] is not None else 0.0
    except Exception:
        pass
    m = re.search(r"([0-9]+\.?[0-9]*)", raw)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return 0.0

def mutate_blueprint(blueprint: Dict[str, Any], problem: str) -> Dict[str, Any]:
    bp_text = json.dumps(blueprint, ensure_ascii=False, indent=2)
    prompt = f"""
Improve the following blueprint for the user task: "{problem}".
Make the rules clearer, add an example of the exact output format, shorten and strengthen the objective,
and produce the improved blueprint as a single JSON object with the same fields (name, objective,
reasoning_style, rules, output_format).

Original blueprint:
{bp_text}
"""
    raw = llm(prompt, model="gpt-4o-mini", max_retries=1, temperature=0.5)
    logger.info("mutate_blueprint raw (trunc): %s", str(raw)[:800])
    candidate = extract_first_json(raw)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            logger.warning("Couldn't parse mutated blueprint JSON")
    improved = dict(blueprint)
    improved.setdefault("rules", [])
    improved["rules"] = improved["rules"] + ["Prefer concise outputs", "Return strictly the requested format"]
    if not improved.get("output_format"):
        improved["output_format"] = "python script"
    return improved

# ----------------------
# /run endpoint: returns only code/plain text (no JSON) on success
# ----------------------
@app.post("/run")
async def run_code_only(request: Request):
    """
    POST JSON: {"problem":"..."}
    On success: returns 200 OK with Content-Type: text/plain and a body that is ONLY the paste-ready code/text
    On failure: returns JSON error (HTTP 4xx/5xx)
    """
    try:
        content = await request.json()
        problem = (content.get("problem") or "").strip()
    except Exception as e:
        logger.exception("Invalid request body to /run")
        return JSONResponse(status_code=400, content={"error": "invalid_request", "detail": str(e)})

    if not problem:
        return JSONResponse(status_code=400, content={"error": "empty_problem"})

    try:
        # 1) blueprints
        blueprints = generate_blueprints(problem)
        for b in blueprints:
            if "output_format" in b:
                b["output_format"] = _normalize_output_format(b.get("output_format", ""))

        # 2) run agents and collect candidates
        candidates: List[Dict[str, Any]] = []
        for bp in blueprints:
            try:
                raw_output = run_agent(bp, problem) or ""
                cleaned = _strip_code_fence(raw_output)
                parsed_output = None
                ofmt_lower = (bp.get("output_format") or "").lower()
                if "json" in ofmt_lower:
                    parsed_output = _try_parse_json_from_text(cleaned)
                score = score_response(problem, cleaned)
                candidates.append({
                    "blueprint": bp,
                    "output": cleaned,
                    "parsed_output": parsed_output,
                    "score": score
                })
            except Exception as e:
                logger.exception("Agent run failed for blueprint %s", bp.get("name"))
                candidates.append({
                    "blueprint": bp,
                    "output": "",
                    "parsed_output": None,
                    "score": 0.0,
                    "error": str(e)
                })

        # 3) pick best candidate (by score) and mutate
        best = max(candidates, key=lambda x: x.get("score", 0.0)) if candidates else None
        mutated = None
        if best:
            mutated_bp = mutate_blueprint(best.get("blueprint", {}), problem)
            mutated_bp["output_format"] = _normalize_output_format(mutated_bp.get("output_format", ""))
            # run mutated blueprint to get final code (prefer to produce code from mutated blueprint)
            final_raw = run_agent(mutated_bp, problem)
            final_output = _strip_code_fence(final_raw) if final_raw else (best.get("output") or "")
            # Return only the code/plain text as text/plain
            return PlainTextResponse(content=final_output, status_code=200)
        else:
            # No candidates produced anything - return empty text
            return PlainTextResponse(content="", status_code=200)

    except Exception as e:
        logger.exception("Unexpected error in /run pipeline")
        return JSONResponse(status_code=500, content={"error": "pipeline_error", "detail": str(e)})

# ----------------------
# health endpoint (JSON is ok here)
# ----------------------
@app.get("/health")
def health():
    return {"status": "ok", "openai_configured": client is not None}

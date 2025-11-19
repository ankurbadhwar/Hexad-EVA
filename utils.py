# utils.py
import os
import re
from typing import List

from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Create a .env file with GEMINI_API_KEY=your_real_key"
    )

genai.configure(api_key=GEMINI_API_KEY)


def sanitize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize_words(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def postprocess_text(
    text: str,
    max_words: int | None = None,
    strip_lines: bool = True,
) -> str:
    if strip_lines:
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    words = text.split()
    if max_words is not None and len(words) > max_words:
        words = words[:max_words]
        text = " ".join(words)
    return text


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 300) -> str:
    """
    Call Gemini using google-generativeai.
    Uses GEMINI_API_KEY and GEMINI_MODEL from .env.
    """
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        },
    )
    # Gemini returns a rich object; .text gives final string
    return response.text

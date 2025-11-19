# data_loader.py
import os
import json
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "testcases.json")


def load_testcases() -> List[Dict[str, Any]]:
    if not os.path.exists(DATA_PATH):
        return [{
            "id": "default_1",
            "input": "This is a sample lecture about AI agents that build other agents.",
            "gold": "AI that autonomously designs agents; Evaluates performance; Evolves better versions.",
            "target_words": 60,
        }]
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

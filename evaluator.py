# evaluator.py
from typing import Dict, Any, List

from utils import tokenize_words


def evaluate_agent(agent: Dict[str, Any], testcases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    testcases: list of {"id": ..., "input": ..., "gold": ..., "target_words": int}
    """
    per_case = []
    scores = []

    run = agent["run"]

    for tc in testcases:
        inp = tc["input"]
        gold = tc["gold"]

        pred = run(inp)

        pred_tokens = set(tokenize_words(pred))
        gold_tokens = set(tokenize_words(gold))

        if gold_tokens:
            overlap_score = len(pred_tokens & gold_tokens) / max(1, len(gold_tokens))
        else:
            overlap_score = 0.0

        pred_words = tokenize_words(pred)
        target_words = tc.get("target_words", 60)
        len_ratio = len(pred_words) / max(1, target_words)
        len_penalty = max(0.0, 1.0 - abs(len_ratio - 1.0))

        final_score = 0.7 * overlap_score + 0.3 * len_penalty

        per_case.append({
            "input_id": tc["id"],
            "input": inp,
            "gold": gold,
            "pred": pred,
            "overlap_score": round(overlap_score, 3),
            "len_penalty": round(len_penalty, 3),
            "final_score": round(final_score, 3),
            "pred_word_count": len(pred_words),
        })
        scores.append(final_score)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "avg_score": round(avg_score, 3),
        "cases": per_case,
    }

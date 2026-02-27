"""LLM-as-Judge: 5-criteria structured evaluation of model outputs."""
from __future__ import annotations

import json
from typing import Any, Dict

from shared.providers import BaseLLM


JUDGE_CRITERIA = [
    "correctness",
    "completeness",
    "factuality",
    "structure",
    "hallucination_resistance",
]

JUDGE_PROMPT = """You are an expert evaluator. Score the model's generated answer against the reference answer on 5 criteria, each from 1 to 10.

Question: {question}

Generated Answer:
{generated}

Reference Answer:
{reference}

Return ONLY a JSON object with this exact format:
{{
  "correctness": {{"score": <1-10>, "reason": "<brief explanation>"}},
  "completeness": {{"score": <1-10>, "reason": "<brief explanation>"}},
  "factuality": {{"score": <1-10>, "reason": "<brief explanation>"}},
  "structure": {{"score": <1-10>, "reason": "<brief explanation>"}},
  "hallucination_resistance": {{"score": <1-10>, "reason": "<brief explanation>"}}
}}

Scoring guidelines:
- correctness: How accurate is the generated answer compared to the reference?
- completeness: Does the generated answer cover all key points from the reference?
- factuality: Are all stated facts verifiable and correct?
- structure: Is the answer well-organized and clearly written?
- hallucination_resistance: Does the answer avoid fabricating information not in the reference?"""


def _zero_scores() -> Dict[str, Dict[str, Any]]:
    """Return a default zero-score result for all criteria."""
    return {
        c: {"score": 0, "reason": "Parse failure — could not extract scores from judge response."}
        for c in JUDGE_CRITERIA
    }


def _clamp_score(value: Any) -> int:
    """Clamp a score to [1, 10] range."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 0
    return max(1, min(10, v))


async def llm_judge(
    question: str,
    generated: str,
    reference: str,
    llm: BaseLLM,
) -> Dict[str, Any]:
    """Run LLM-as-Judge evaluation on a single generated output.

    Args:
        question: The original question / instruction.
        generated: Model-generated answer.
        reference: Ground truth reference answer.
        llm: A BaseLLM provider for the judge calls.

    Returns:
        Dict with keys for each criterion, each containing {score, reason}.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        generated=generated,
        reference=reference,
    )

    response = await llm.chat(
        [{"role": "user", "content": prompt}],
    )

    try:
        parsed = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        return _zero_scores()

    result: Dict[str, Any] = {}
    for criterion in JUDGE_CRITERIA:
        entry = parsed.get(criterion, {})
        if isinstance(entry, dict):
            result[criterion] = {
                "score": _clamp_score(entry.get("score", 0)),
                "reason": str(entry.get("reason", "")),
            }
        else:
            result[criterion] = {"score": 0, "reason": "Missing criterion in judge response."}

    return result

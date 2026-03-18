from __future__ import annotations

from typing import Dict, Optional


MODEL_COST_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50 / 1e6, "output": 10.0 / 1e6},
    "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    "o3-mini": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
}


def get_cost_rates(model_id: str) -> Optional[Dict[str, float]]:
    return MODEL_COST_TABLE.get(model_id)


def estimate_cost_usd(
    model_id: str,
    usage: Optional[Dict[str, int]],
    *,
    unknown_as_zero: bool = False,
) -> Optional[float]:
    if not usage:
        return 0.0 if unknown_as_zero else None

    rates = get_cost_rates(model_id)
    if rates is None:
        return 0.0 if unknown_as_zero else None

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    return prompt_tokens * rates["input"] + completion_tokens * rates["output"]

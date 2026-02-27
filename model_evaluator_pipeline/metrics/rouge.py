"""ROUGE scoring for model outputs vs ground truth references."""
from __future__ import annotations

import asyncio
from typing import Dict


async def compute_rouge(generated: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores.

    Args:
        generated: Model-generated text.
        reference: Ground truth reference text.

    Returns:
        Dict with keys rouge1, rouge2, rougeL, rougeLsum — each a float in [0, 1].
    """
    import evaluate

    scorer = evaluate.load("rouge")

    # evaluate.compute is synchronous — offload to thread to stay async
    result = await asyncio.to_thread(
        scorer.compute,
        predictions=[generated],
        references=[reference],
    )

    return {
        "rouge1": float(result["rouge1"]),
        "rouge2": float(result["rouge2"]),
        "rougeL": float(result["rougeL"]),
        "rougeLsum": float(result["rougeLsum"]),
    }

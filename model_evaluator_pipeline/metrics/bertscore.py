"""BERTScore for semantic similarity between model outputs and references."""
from __future__ import annotations

import asyncio
from typing import Dict

import bert_score


async def compute_bertscore(
    generated: str,
    reference: str,
    model_type: str = "roberta-large",
) -> Dict[str, float]:
    """Compute BERTScore precision, recall, and F1.

    Args:
        generated: Model-generated text.
        reference: Ground truth reference text.
        model_type: The model to use for BERTScore embeddings.

    Returns:
        Dict with keys bert_precision, bert_recall, bert_f1 — each a float in [0, 1].
    """
    # bert_score.score is synchronous and GPU-heavy — offload to thread
    precision, recall, f1 = await asyncio.to_thread(
        bert_score.score,
        cands=[generated],
        refs=[reference],
        model_type=model_type,
        verbose=False,
    )

    return {
        "bert_precision": float(precision[0]),
        "bert_recall": float(recall[0]),
        "bert_f1": float(f1[0]),
    }

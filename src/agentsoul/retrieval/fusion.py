"""
Fusion strategies for hybrid retrieval (vector + BM25).

Provides base concatenation fusion and Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def base_fusion(
    vec_results: List[Dict[str, Any]],
    bm25_results: List[Tuple[str, float]],
    top_k: int = 20,
) -> List[str]:
    """Simple rank-based fusion: vector results first, then BM25 results (deduped).

    Args:
        vec_results: List of dicts with ``doc_id`` and ``score`` keys (higher=better).
        bm25_results: List of ``(doc_id, score)`` tuples (higher=better).
        top_k: Maximum number of fused results to return.

    Returns:
        List of top doc_ids after fusion.
    """
    vec_sorted = sorted(vec_results, key=lambda d: d["score"], reverse=True)
    bm25_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)

    vec_ids = {d["doc_id"] for d in vec_sorted}
    bm25_deduped = [
        {"doc_id": doc_id, "score": score}
        for doc_id, score in bm25_sorted
        if doc_id not in vec_ids
    ]

    fused = vec_sorted + bm25_deduped
    return [d["doc_id"] for d in fused[:top_k]]


def reciprocal_rank_fusion(
    vec_results: List[Dict[str, Any]],
    bm25_results: List[Tuple[str, float]],
    top_k: int = 20,
    rrf_k: int = 60,
) -> Tuple[List[str], List[float]]:
    """Reciprocal Rank Fusion combining vector and BM25 rankings.

    Args:
        vec_results: List of dicts with ``doc_id`` and ``score`` keys (higher=better).
        bm25_results: List of ``(doc_id, score)`` tuples (higher=better).
        top_k: Maximum number of fused results to return.
        rrf_k: RRF constant (default 60). Higher values smooth rank differences.

    Returns:
        Tuple of (doc_ids, fused_scores) sorted by fused score descending.
    """
    vec_sorted = sorted(vec_results, key=lambda d: d["score"], reverse=True)
    bm25_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)

    vec_rank = {d["doc_id"]: i + 1 for i, d in enumerate(vec_sorted)}
    bm25_rank = {doc_id: i + 1 for i, (doc_id, _) in enumerate(bm25_sorted)}

    candidates = set(vec_rank.keys()) | set(bm25_rank.keys())
    scores: Dict[str, float] = {}
    for doc_id in candidates:
        r1 = vec_rank.get(doc_id, 10**9)
        r2 = bm25_rank.get(doc_id, 10**9)
        scores[doc_id] = 1.0 / (rrf_k + r1) + 1.0 / (rrf_k + r2)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    doc_ids = [doc_id for doc_id, _ in fused]
    fused_scores = [scores[doc_id] for doc_id in doc_ids]
    return doc_ids, fused_scores


def build_llm_context(
    doc_ids: List[str],
    texts_by_id: Dict[str, str],
    metadatas_by_id: Dict[str, Dict] = None,
) -> str:
    """Build a single context string from fused document IDs.

    Args:
        doc_ids: Ordered list of document IDs to include.
        texts_by_id: Mapping of doc_id to document text.
        metadatas_by_id: Optional mapping of doc_id to metadata dict.

    Returns:
        Concatenated context string with ``---`` separators.
    """
    parts: List[str] = []
    for doc_id in doc_ids:
        text = texts_by_id.get(doc_id)
        if text is None:
            continue
        parts.append(text.strip())

    context = "\n\n---\n\n".join(parts)
    # Collapse excessive blank lines (3+) but preserve paragraph breaks
    import re as _re
    return _re.sub(r'\n{3,}', '\n\n', context)

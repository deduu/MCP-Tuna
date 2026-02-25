"""
Production-hardened BM25 text search service.

Uses JSON serialization (not pickle) for security.
Thread-safe with input validation and proper logging.

Requires: pip install rank-bm25  (or agentsoul[retrieval])
"""

import heapq
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .tokenizer import simple_tokenize

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is required for BM25Service. "
        "Install it with: pip install rank-bm25  "
        "or: pip install agentsoul[retrieval]"
    )

logger = logging.getLogger(__name__)


@dataclass
class BM25Pack:
    """Serializable BM25 index data."""
    doc_ids: List[str]
    texts: List[str]
    metadatas: List[Dict[str, Any]]
    tokenized_corpus: List[List[str]]


@dataclass
class BM25Runtime:
    """In-memory BM25 runtime with prebuilt index."""
    pack: BM25Pack
    bm25: BM25Okapi
    id2idx: Dict[str, int]


class BM25Service:
    """Thread-safe BM25 text search service with JSON persistence.

    Args:
        index_path: Path to the BM25 JSON index file.
    """

    def __init__(self, index_path: str):
        self.path = index_path
        self._lock = threading.Lock()
        self.runtime = self._load_runtime(self.path)

    def _load_runtime(self, path: str) -> BM25Runtime:
        """Load BM25 runtime from JSON index file."""
        if not os.path.exists(path):
            logger.warning("[BM25] Index file not found at %s. Creating empty runtime.", path)
            return self._empty_runtime()

        try:
            logger.info("[BM25] Loading BM25 index from %s", path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            pack = BM25Pack(
                doc_ids=data["doc_ids"],
                texts=data["texts"],
                metadatas=data["metadatas"],
                tokenized_corpus=data["tokenized_corpus"],
            )

            if not pack.tokenized_corpus:
                logger.warning("[BM25] Loaded index has empty corpus.")
                return self._empty_runtime()

            bm25 = BM25Okapi(pack.tokenized_corpus)
            id2idx = {did: i for i, did in enumerate(pack.doc_ids)}
            logger.info("[BM25] Loaded %d documents.", len(pack.doc_ids))
            return BM25Runtime(pack=pack, bm25=bm25, id2idx=id2idx)

        except Exception as e:
            logger.error("[BM25] Failed to load index from %s: %s", path, e)
            return self._empty_runtime()

    @staticmethod
    def _empty_runtime() -> BM25Runtime:
        """Create a safe empty BM25 runtime (no searchable documents)."""
        empty_pack = BM25Pack(doc_ids=[], texts=[], metadatas=[], tokenized_corpus=[])
        empty_bm25 = BM25Okapi([[]])
        return BM25Runtime(pack=empty_pack, bm25=empty_bm25, id2idx={})

    def bm25_search(self, queries: List[str], k: int = 50) -> List[Tuple[str, float]]:
        """Search top-k documents using BM25.

        Args:
            queries: List of query strings.
            k: Maximum number of results to return.

        Returns:
            List of (doc_id, score) tuples sorted by score descending.
        """
        if not queries or not isinstance(queries, list):
            logger.warning("[BM25] Invalid queries input: %s", type(queries))
            return []

        with self._lock:
            runtime = self.runtime

            if not runtime.pack.doc_ids:
                logger.warning("[BM25] Search skipped: empty index.")
                return []

            retrieved_results = []
            retrieved_set: set = set()

            for query in queries:
                if not isinstance(query, str) or not query.strip():
                    continue
                scores = runtime.bm25.get_scores(simple_tokenize(query))
                top_indices = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])
                for i in top_indices:
                    doc_id = runtime.pack.doc_ids[i]
                    score = float(scores[i])
                    if doc_id not in retrieved_set:
                        retrieved_set.add(doc_id)
                        retrieved_results.append((doc_id, score))

        return sorted(retrieved_results, key=lambda x: x[1], reverse=True)[:k]

    def reload_index(self, path: str = None) -> None:
        """Reload BM25 index from disk. Thread-safe.

        Args:
            path: Optional override path. Uses original path if not provided.
        """
        path = path or self.path
        new_runtime = self._load_runtime(path)
        with self._lock:
            self.runtime = new_runtime
        logger.info("[BM25] Index reloaded from %s", path)

    @staticmethod
    def build_index(
        doc_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        out_path: str,
    ) -> str:
        """Build a BM25 index and save to JSON.

        Args:
            doc_ids: Document identifiers.
            texts: Document texts.
            metadatas: Document metadata dicts.
            out_path: Path to write the JSON index file.

        Returns:
            The output path.
        """
        tokenized_corpus = [simple_tokenize(text) for text in texts]

        data = {
            "doc_ids": doc_ids,
            "texts": texts,
            "metadatas": metadatas,
            "tokenized_corpus": tokenized_corpus,
        }

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        logger.info("[BM25] Saved %d chunks to %s", len(texts), out_path)
        return out_path

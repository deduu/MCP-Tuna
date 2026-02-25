"""
RAG Tool Service — wraps FAISS+BM25 hybrid search as LLM-callable tools.

Extends agentsoul's ToolService so the agent can call retrieval tools during
its agentic loop.

Requires: pip install agentsoul[retrieval]
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from agentsoul.tools.service import ToolService

from .bm25_service import BM25Service
from .faiss_service import FAISSService
from .fusion import base_fusion, build_llm_context, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class RAGToolService(ToolService):
    """Tool service that registers hybrid FAISS+BM25 retrieval as LLM tools.

    Args:
        faiss_service: An initialized FAISSService instance.
        bm25_service: An initialized BM25Service instance.
        fusion_mode: Fusion strategy — ``"base"`` (concatenation) or ``"rrf"``
            (Reciprocal Rank Fusion). Defaults to ``"base"``.
        rrf_k: RRF constant (only used when ``fusion_mode="rrf"``).
    """

    def __init__(
        self,
        faiss_service: FAISSService,
        bm25_service: BM25Service,
        fusion_mode: str = "base",
        rrf_k: int = 60,
    ):
        super().__init__()
        self.faiss_service = faiss_service
        self.bm25_service = bm25_service
        self.fusion_mode = fusion_mode
        self.rrf_k = rrf_k

        self._register_tools()

    def _register_tools(self):
        """Register retrieval tools that the LLM agent can call."""

        @self.tool(
            name="search_knowledge_base",
            description=(
                "Search the knowledge base using hybrid retrieval (vector similarity + BM25 text matching). "
                "Returns relevant document chunks as context for answering questions. "
                "Use this tool when you need to look up definitions, rules, frameworks, or any domain knowledge."
            ),
        )
        def search_knowledge_base(
            queries: str,
            top_k: int = 20,
        ) -> str:
            """Search the knowledge base with hybrid retrieval.

            Args:
                queries: The search query string. Use the user's original question wording.
                top_k: Maximum number of document chunks to return.
            """
            # Normalize to list
            if isinstance(queries, str):
                query_list = [queries]
            elif isinstance(queries, list):
                query_list = queries
            else:
                return "Error: queries must be a string or list of strings."

            return self._hybrid_search(query_list, top_k=top_k)

    def _hybrid_search(self, queries: List[str], top_k: int = 20) -> str:
        """Execute hybrid FAISS + BM25 search with fusion.

        Args:
            queries: List of query strings.
            top_k: Final number of results after fusion.

        Returns:
            Context string with fused results.
        """
        k_vector = top_k
        k_bm25 = top_k

        # Vector search
        vec_results, _ = self.faiss_service.search_with_ranks(queries, top_k=k_vector)
        logger.info("Vector search returned %d chunks.", len(vec_results))

        # BM25 search
        bm25_results = self.bm25_service.bm25_search(queries=queries, k=k_bm25)
        logger.info("BM25 search returned %d chunks.", len(bm25_results))

        # Fusion
        if self.fusion_mode == "rrf":
            fused_ids, fused_scores = reciprocal_rank_fusion(
                vec_results, bm25_results, top_k=top_k, rrf_k=self.rrf_k
            )
        else:
            fused_ids = base_fusion(vec_results, bm25_results, top_k=top_k)

        # Build text lookup from both sources
        texts_by_id = self._build_texts_lookup(vec_results, bm25_results)

        context = build_llm_context(fused_ids, texts_by_id)

        if not context.strip():
            return "No relevant context found for the given query."

        logger.info("Hybrid search fused %d unique chunks into context.", len(fused_ids))
        return context

    def _build_texts_lookup(
        self,
        vec_results: List[Dict[str, Any]],
        bm25_results: List[Tuple[str, float]],
    ) -> Dict[str, str]:
        """Build a doc_id -> text mapping from both retrieval sources.

        Vector results carry text directly. For BM25 results, we look up text
        from the BM25 runtime pack.
        """
        texts: Dict[str, str] = {}

        # From vector results
        for item in vec_results:
            doc_id = item["doc_id"]
            if doc_id not in texts:
                texts[doc_id] = item.get("text", "")

        # From BM25 runtime (hold lock for thread-safe access)
        with self.bm25_service._lock:
            runtime = self.bm25_service.runtime
            for doc_id, _ in bm25_results:
                if doc_id not in texts:
                    idx = runtime.id2idx.get(doc_id)
                    if idx is not None:
                        texts[doc_id] = runtime.pack.texts[idx]

        return texts

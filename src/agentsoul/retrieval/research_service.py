"""
Research Tool Service — concurrent multi-source retrieval.

Fans out to up to three sources concurrently:
  1. Web search (via a provided async callable)
  2. Document retrieval (FAISS + BM25 hybrid via RAGToolService)
  3. Memory / chat history search (via BaseMemory)

Each source is **optional**.  Missing or failing sources are silently
skipped, so the tool always returns whatever information is available.

Extends agentsoul's ToolService so the agent can call it during its
agentic loop.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from agentsoul.tools.service import ToolService

logger = logging.getLogger(__name__)


class ResearchToolService(ToolService):
    """Tool service that registers a concurrent multi-source research tool.

    All three sources are optional — pass only what you have:

    Args:
        rag_service: Optional ``RAGToolService`` (or any object that
            exposes ``_hybrid_search(queries, top_k=…) -> str``).
        memory: Optional ``BaseMemory`` instance for chat-history /
            long-term memory search.
        web_search_fn: Optional async callable
            ``(query: str, **kw) -> dict | str`` for web search.
            Typically ``web_service.deep_search`` or ``web_service.search``.

    Example::

        research = ResearchToolService(
            rag_service=rag,
            memory=composite_memory,
            web_search_fn=web_service.deep_search,
        )

        composite = CompositeToolService()
        composite.register_service("rag", rag)
        composite.register_service("research", research)
    """

    def __init__(
        self,
        rag_service: Optional[Any] = None,
        memory: Optional[Any] = None,
        web_search_fn: Optional[Callable[..., Coroutine]] = None,
    ):
        super().__init__()
        self.rag_service = rag_service
        self.memory = memory
        self.web_search_fn = web_search_fn

        self._register_tools()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_tools(self):
        sources = []
        if self.web_search_fn is not None:
            sources.append("web search")
        if self.rag_service is not None:
            sources.append("knowledge base documents (FAISS + BM25)")
        if self.memory is not None:
            sources.append("conversation history / memory")

        source_list = ", ".join(sources) if sources else "no sources configured"

        @self.tool(
            name="deep_research",
            description=(
                "Research a topic by concurrently consulting multiple information sources: "
                f"{source_list}. "
                "Returns combined context from all available sources.\n\n"
                "Use this tool when you need comprehensive information to answer a "
                "complex question. Only sources that are configured and available "
                "will be consulted — missing sources are silently skipped."
            ),
        )
        async def deep_research(query: str, top_k: int = 10) -> str:
            """Research across all available sources concurrently.

            Args:
                query: The research question or topic.
                top_k: Maximum results per source (default 10).
            """
            return await self._research(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Core orchestration
    # ------------------------------------------------------------------

    async def _research(self, query: str, top_k: int = 10) -> str:
        """Fan out to all configured sources and combine results."""
        tasks: Dict[str, asyncio.Task] = {}

        if self.web_search_fn is not None:
            tasks["web"] = self._search_web(query)
        if self.rag_service is not None:
            tasks["documents"] = self._search_documents(query, top_k=top_k)
        if self.memory is not None:
            tasks["memory"] = self._search_memory(query, top_k=top_k)

        if not tasks:
            return "No research sources are configured."

        keys = list(tasks.keys())
        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )

        sections: List[str] = []
        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                logger.warning("Research source '%s' failed: %s", key, result)
                continue
            if not result:
                continue
            sections.append(result)

        if not sections:
            return "No relevant information found from any source."

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Individual source handlers
    # ------------------------------------------------------------------

    async def _search_web(self, query: str) -> str:
        """Run web search and format results."""
        try:
            result = await self.web_search_fn(query)
        except Exception as e:
            logger.warning("Web search error: %s", e)
            return ""

        if isinstance(result, str):
            return f"## Web Search Results\n\n{result}" if result.strip() else ""

        if not isinstance(result, dict):
            return ""

        if not result.get("success", True):
            logger.warning("Web search failed: %s", result.get("error", "unknown"))
            return ""

        # deep_search format — sources with chunks
        sources = result.get("sources", [])
        if sources:
            parts = []
            for src in sources:
                title = src.get("title", "Untitled")
                url = src.get("url", "")
                chunks = src.get("chunks", [])
                content = "\n".join(chunks) if chunks else src.get("snippet", "")
                parts.append(f"### {title}\nURL: {url}\n{content}")
            return "## Web Search Results\n\n" + "\n\n---\n\n".join(parts)

        # simple search format — results list
        results_list = result.get("results", [])
        if results_list:
            parts = []
            for r in results_list:
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                parts.append(f"### {title}\nURL: {url}\n{snippet}")
            return "## Web Search Results\n\n" + "\n\n---\n\n".join(parts)

        return ""

    async def _search_documents(self, query: str, top_k: int = 10) -> str:
        """Run hybrid RAG search and format results."""
        try:
            # _hybrid_search uses thread-locked BM25 + FAISS — run in thread
            result = await asyncio.to_thread(
                self.rag_service._hybrid_search, [query], top_k=top_k
            )
        except Exception as e:
            logger.warning("Document retrieval error: %s", e)
            return ""

        if not result or result == "No relevant context found for the given query.":
            return ""

        return f"## Knowledge Base Results\n\n{result}"

    async def _search_memory(self, query: str, top_k: int = 5) -> str:
        """Search memory / chat history and format results."""
        try:
            results = await self.memory.search(query, top_k=top_k)
        except Exception as e:
            logger.warning("Memory search error: %s", e)
            return ""

        if not results:
            return ""

        parts = []
        for r in results:
            content = r.get("content", "")
            if not content:
                continue
            meta = r.get("metadata", {})
            user_ctx = meta.get("user_context", "")
            if user_ctx:
                parts.append(f"[Context: {user_ctx}]\n{content}")
            else:
                parts.append(content)

        if not parts:
            return ""

        return "## Conversation History\n\n" + "\n\n---\n\n".join(parts)

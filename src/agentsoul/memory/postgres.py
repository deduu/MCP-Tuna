"""Long-term vector memory backed by a PostgreSQL + pgvector store.

Wraps any store object that exposes:
    - ``available: bool``
    - ``async save(content: str) -> int``
    - ``async search(query: str, *, limit: int, threshold: float) -> list[dict]``

The adapter itself has **no hard dependencies** — ``asyncpg``, ``pgvector``,
and the embedding client are the store's responsibility.  This keeps the
framework import lightweight while still supporting Postgres persistence.

Requires (for the store): ``pip install asyncpg pgvector``
    (or ``pip install agentsoul[postgres]``)
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from agentsoul.core.models import Message, MessageRole
from .base import BaseMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duck-typed protocol for the underlying store
# ---------------------------------------------------------------------------

@runtime_checkable
class VectorStore(Protocol):
    """Structural interface that any compatible store must satisfy."""

    available: bool

    async def save(self, content: str) -> int: ...

    async def search(
        self, query: str, *, limit: int = 5, threshold: float = 0.3
    ) -> list: ...


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class PostgresMemory(BaseMemory):
    """Long-term vector memory adapter for PostgreSQL + pgvector stores.

    Accepts any store implementing the :class:`VectorStore` protocol
    (``save``, ``search``, ``available``).  Does **not** manage the store
    lifecycle — the caller is responsible for ``init()`` / ``close()``.

    Args:
        agent_id: Identifier for this agent (used by BaseMemory).
        store: A ``VectorStore``-compatible object (e.g.
            ``examples.backend.core.memory_store.memory_store``).
        team_id: Optional team scope.
        similarity_threshold: Minimum similarity for search results.
        max_results: Maximum memories to retrieve per query.
        auto_save_assistant: If True, ``add_messages`` automatically
            stores assistant responses for future recall.
    """

    def __init__(
        self,
        agent_id: str,
        store: Any,
        team_id: Optional[str] = None,
        similarity_threshold: float = 0.3,
        max_results: int = 5,
        auto_save_assistant: bool = True,
    ):
        super().__init__(agent_id, team_id)
        self._store = store
        self._threshold = similarity_threshold
        self._max_results = max_results
        self._auto_save_assistant = auto_save_assistant

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return getattr(self._store, "available", False)

    # ------------------------------------------------------------------
    # BaseMemory: automatic context injection
    # ------------------------------------------------------------------

    async def get_context_messages(
        self, query: str, max_tokens: int = 2000
    ) -> List[Message]:
        if not self.available:
            return []

        try:
            results = await self._store.search(
                query, limit=self._max_results, threshold=self._threshold
            )
        except Exception as e:
            logger.warning("PostgresMemory search failed: %s", e)
            return []

        if not results:
            return []

        # Build context respecting token budget (4-char heuristic)
        lines: List[str] = []
        token_count = 0
        for r in results:
            text = r.get("content", "")
            tokens = len(text) // 4
            if token_count + tokens > max_tokens:
                break
            similarity = r.get("similarity", 0)
            lines.append(f"[similarity={similarity:.2f}] {text}")
            token_count += tokens

        if not lines:
            return []

        context_text = "\n".join(lines)
        return [
            Message(
                MessageRole.SYSTEM,
                f"## Relevant Memory\n{context_text}",
            )
        ]

    # ------------------------------------------------------------------
    # BaseMemory: store messages after each run
    # ------------------------------------------------------------------

    async def add_messages(self, messages: List[Message]) -> None:
        if not self.available or not self._auto_save_assistant:
            return

        last_user_content = ""
        for msg in messages:
            if msg.role == MessageRole.USER:
                last_user_content = msg.content
            elif msg.role == MessageRole.ASSISTANT and msg.content.strip():
                # Prefix with user question for richer embedding retrieval
                content = msg.content
                if last_user_content:
                    content = (
                        f"Q: {last_user_content[:200]}\n"
                        f"A: {msg.content}"
                    )
                try:
                    await self._store.save(content)
                except Exception as e:
                    logger.warning("PostgresMemory failed to save message: %s", e)

    # ------------------------------------------------------------------
    # BaseMemory: explicit store / search / clear
    # ------------------------------------------------------------------

    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        scope: str = "private",
    ) -> None:
        if not self.available:
            return
        try:
            await self._store.save(content)
        except Exception as e:
            logger.warning("PostgresMemory store failed: %s", e)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        scope: str = "private",
    ) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        try:
            results = await self._store.search(
                query, limit=top_k, threshold=self._threshold
            )
            # Normalize to BaseMemory format {content, metadata, score}.
            # MemoryStore returns similarity (higher=better).
            # CompositeMemory sorts by score ascending (lower=better).
            # Convert: score = 1 - similarity.
            return [
                {
                    "content": r.get("content", ""),
                    "metadata": {
                        "id": r.get("id"),
                        "created_at": r.get("created_at"),
                        "scope": scope,
                    },
                    "score": 1.0 - r.get("similarity", 0),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("PostgresMemory search failed: %s", e)
            return []

    async def clear(self) -> None:
        logger.warning(
            "PostgresMemory.clear() is a no-op. "
            "Use direct database operations to clear the memories table."
        )

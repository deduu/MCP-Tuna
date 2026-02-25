import asyncio
from typing import List, Dict, Any, Optional

from agentsoul.core.models import Message
from .base import BaseMemory


class CompositeMemory(BaseMemory):
    """Combines multiple memory backends into a single interface.

    Typical usage: CompositeMemory([ChromaMemory(...), ConversationMemory(...)])
    - get_context_messages merges results from all backends (long-term first)
    - add_messages delegates to all backends
    - search aggregates and sorts by score
    """

    def __init__(
        self,
        backends: List[BaseMemory],
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ):
        _agent_id = agent_id or (backends[0].agent_id if backends else "composite")
        _team_id = team_id or (backends[0].team_id if backends else None)
        super().__init__(_agent_id, _team_id)
        self.backends = backends

    async def get_context_messages(
        self, query: str, max_tokens: int = 2000
    ) -> List[Message]:
        # Distribute token budget evenly, then collect concurrently
        per_backend = max_tokens // max(len(self.backends), 1)

        results = await asyncio.gather(
            *(b.get_context_messages(query, max_tokens=per_backend) for b in self.backends)
        )
        all_messages: List[Message] = []
        for msgs in results:
            all_messages.extend(msgs)
        return all_messages

    async def add_messages(self, messages: List[Message]) -> None:
        await asyncio.gather(
            *(b.add_messages(messages) for b in self.backends)
        )

    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        scope: str = "private",
    ) -> None:
        await asyncio.gather(
            *(b.store(content, metadata, scope) for b in self.backends)
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        scope: str = "private",
    ) -> List[Dict[str, Any]]:
        results_per_backend = await asyncio.gather(
            *(b.search(query, top_k=top_k, scope=scope) for b in self.backends)
        )
        all_results: List[Dict[str, Any]] = []
        for results in results_per_backend:
            all_results.extend(results)

        # Sort by score (lower = better) and limit
        all_results.sort(key=lambda x: x.get("score", float("inf")))
        return all_results[:top_k]

    async def clear(self) -> None:
        await asyncio.gather(
            *(b.clear() for b in self.backends)
        )

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from agentsoul.core.models import Message


class BaseMemory(ABC):
    """Abstract base class for memory backends.

    Every memory backend must implement:
    - get_context_messages: retrieve relevant context to inject into conversation
    - add_messages: store messages from a completed run

    Optional methods (no-op by default):
    - store: explicitly store a fact
    - search: semantic search over stored content
    - clear: wipe all stored memory
    """

    def __init__(self, agent_id: str, team_id: Optional[str] = None):
        self.agent_id = agent_id
        self.team_id = team_id

    @abstractmethod
    async def get_context_messages(
        self, query: str, max_tokens: int = 2000
    ) -> List[Message]:
        """Retrieve context messages to inject into the LLM conversation.

        Args:
            query: The current user query for relevance matching.
            max_tokens: Approximate token budget for returned context.

        Returns:
            List of Message objects to inject between system prompt and user message.
        """
        ...

    @abstractmethod
    async def add_messages(self, messages: List[Message]) -> None:
        """Store messages from a completed agent run.

        Args:
            messages: The full message list from the run.
        """
        ...

    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        scope: str = "private",
    ) -> None:
        """Explicitly store a fact or piece of information.

        Args:
            content: The text content to store.
            metadata: Optional metadata to attach.
            scope: "private" (agent-only) or "team" (shared).
        """

    async def search(
        self,
        query: str,
        top_k: int = 5,
        scope: str = "private",
    ) -> List[Dict[str, Any]]:
        """Semantic search over stored content.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.
            scope: "private", "team", or "all".

        Returns:
            List of dicts with keys: content, metadata, score.
        """
        return []

    async def clear(self) -> None:
        """Wipe all stored memory for this agent."""

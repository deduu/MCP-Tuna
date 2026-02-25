import inspect
from typing import List, Optional, Callable

from agentsoul.core.models import Message, MessageRole
from .base import BaseMemory


class ConversationMemory(BaseMemory):
    """Short-term in-memory conversation buffer.

    Persists messages across multiple run() calls on the same agent instance.
    Uses a sliding window to keep the buffer bounded.

    Zero external dependencies.
    """

    def __init__(
        self,
        agent_id: str,
        team_id: Optional[str] = None,
        max_messages: int = 50,
        max_tokens: int = 4000,
        summarizer: Optional[Callable[[List[Message]], Message]] = None,
    ):
        super().__init__(agent_id, team_id)
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarizer = summarizer
        self._buffer: List[Message] = []

    async def get_context_messages(
        self, query: str, max_tokens: int = 2000
    ) -> List[Message]:
        if not self._buffer:
            return []

        # Apply token budget (4-chars-per-token heuristic)
        budget = min(max_tokens, self.max_tokens)
        selected: List[Message] = []
        token_count = 0

        # Walk backwards to prefer recent messages, then reverse
        for msg in reversed(self._buffer):
            msg_tokens = len(msg.content) // 4
            if token_count + msg_tokens > budget:
                break
            selected.append(msg)
            token_count += msg_tokens
        selected.reverse()

        if not selected:
            return []

        # Wrap conversation history in a system message
        lines = []
        for msg in selected:
            lines.append(f"[{msg.role.value}]: {msg.content}")
        history_text = "\n".join(lines)

        return [
            Message(
                MessageRole.SYSTEM,
                f"## Previous Conversation History\n{history_text}",
            )
        ]

    async def add_messages(self, messages: List[Message]) -> None:
        # Filter out system messages — agent manages those itself
        filtered = [
            m for m in messages if m.role != MessageRole.SYSTEM
        ]
        self._buffer.extend(filtered)

        # Sliding window truncation
        if len(self._buffer) > self.max_messages:
            overflow = self._buffer[: len(self._buffer) - self.max_messages]
            self._buffer = self._buffer[-self.max_messages :]

            # Summarize overflow if a summarizer is provided
            if self.summarizer and overflow:
                if inspect.iscoroutinefunction(self.summarizer):
                    summary = await self.summarizer(overflow)
                else:
                    summary = self.summarizer(overflow)
                self._buffer.insert(0, summary)

    async def clear(self) -> None:
        self._buffer.clear()

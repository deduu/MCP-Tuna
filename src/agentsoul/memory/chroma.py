import asyncio
import uuid
from typing import List, Dict, Any, Optional

from agentsoul.core.models import Message, MessageRole
from .base import BaseMemory


class ChromaMemory(BaseMemory):
    """Long-term vector memory backed by ChromaDB.

    Maintains two collections per agent:
    - Private: memory_{agent_id} (agent-scoped)
    - Team: memory_team_{team_id} (shared across team agents)

    Requires: pip install chromadb  (or agentsoul[memory])
    """

    def __init__(
        self,
        agent_id: str,
        team_id: Optional[str] = None,
        chroma_client: Optional[Any] = None,
        private_collection_name: Optional[str] = None,
        team_collection_name: Optional[str] = None,
    ):
        super().__init__(agent_id, team_id)

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is required for ChromaMemory. "
                "Install it with: pip install chromadb  "
                "or: pip install agentsoul[memory]"
            )

        self._client = chroma_client or chromadb.Client()

        # Private collection
        self._private_name = private_collection_name or f"memory_{agent_id}"
        self._private = self._client.get_or_create_collection(
            name=self._private_name
        )

        # Team collection (optional)
        self._team = None
        self._team_name = None
        if team_id:
            self._team_name = team_collection_name or f"memory_team_{team_id}"
            self._team = self._client.get_or_create_collection(
                name=self._team_name
            )

    async def get_context_messages(
        self, query: str, max_tokens: int = 2000
    ) -> List[Message]:
        results = await self.search(query, top_k=5, scope="all")
        if not results:
            return []

        # Build context from search results, respecting token budget
        lines = []
        token_count = 0
        for r in results:
            text = r["content"]
            tokens = len(text) // 4
            if token_count + tokens > max_tokens:
                break
            scope_tag = r.get("metadata", {}).get("scope", "private")
            lines.append(f"[{scope_tag}] {text}")
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

    async def add_messages(self, messages: List[Message]) -> None:
        # Store assistant responses; use preceding user message as metadata context
        last_user_content = ""
        for msg in messages:
            if msg.role == MessageRole.USER:
                last_user_content = msg.content
            elif msg.role == MessageRole.ASSISTANT and msg.content.strip():
                await self.store(
                    content=msg.content,
                    metadata={
                        "role": "assistant",
                        "user_context": last_user_content[:500],
                    },
                    scope="private",
                )

    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        scope: str = "private",
    ) -> None:
        meta = dict(metadata or {})
        meta["scope"] = scope
        doc_id = str(uuid.uuid4())

        if scope == "team" and self._team is not None:
            meta["agent_id"] = self.agent_id
            await asyncio.to_thread(
                self._team.add,
                documents=[content],
                metadatas=[meta],
                ids=[doc_id],
            )
        else:
            await asyncio.to_thread(
                self._private.add,
                documents=[content],
                metadatas=[meta],
                ids=[doc_id],
            )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        scope: str = "private",
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if scope in ("private", "all"):
            count = await asyncio.to_thread(self._private.count)
            if count > 0:
                private_results = await asyncio.to_thread(
                    self._private.query,
                    query_texts=[query],
                    n_results=min(top_k, count),
                )
                results.extend(
                    self._parse_chroma_results(private_results, "private")
                )

        if scope in ("team", "all"):
            if self._team is not None:
                team_count = await asyncio.to_thread(self._team.count)
                if team_count > 0:
                    team_results = await asyncio.to_thread(
                        self._team.query,
                        query_texts=[query],
                        n_results=min(top_k, team_count),
                    )
                    results.extend(
                        self._parse_chroma_results(team_results, "team")
                    )

        # Sort by score (lower distance = more relevant)
        results.sort(key=lambda x: x.get("score", float("inf")))
        return results[:top_k]

    async def clear(self) -> None:
        await asyncio.to_thread(self._client.delete_collection, self._private_name)
        self._private = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=self._private_name,
        )
        if self._team is not None and self._team_name:
            await asyncio.to_thread(self._client.delete_collection, self._team_name)
            self._team = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=self._team_name,
            )

    @staticmethod
    def _parse_chroma_results(
        raw: Dict[str, Any], scope: str
    ) -> List[Dict[str, Any]]:
        parsed = []
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            meta = meta or {}
            meta["scope"] = scope
            parsed.append({
                "content": doc,
                "metadata": meta,
                "score": dist,
            })
        return parsed

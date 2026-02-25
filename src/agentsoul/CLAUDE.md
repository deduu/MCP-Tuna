# AgentSoul — Bundled Agent Framework

## This Directory
`src/agentsoul/` is the bundled custom agent framework. It is built into the project wheel
and imported as `agentsoul.*`. Do NOT install it as an external package or import it
as `src.agentsoul`. Do NOT add application-specific logic here.

## Capabilities

| Module | Contents |
|--------|----------|
| `core/` | `AgentSoul` (main agent), `Message`, `ToolStrategy` |
| `strategies/` | `CodeAct` (Python exec), `JSONSchema` (structured tool calls) |
| `providers/` | `BaseLLM` ABC, `OpenAIProvider`, `HuggingFaceProvider` |
| `memory/` | `ConversationMemory`, `ChromaMemory`, `PostgreSQLMemory` |
| `retrieval/` | `FAISSService`, `BM25Service`, `RAGToolService`, `FusionRetriever` |
| `multi/` | `AgentTeam`, `AgentAsTool`, `TaskDecomposer` |
| `server/` | `MCPServer` — HTTP+stdio MCP transport |
| `utils/` | `logger.py` (structured logging), `tracker.py` (execution tracking) |

## Key Rules
- Use `agentsoul.utils.logger` for ALL logging — never `print()` or `logging`
- `AgentSoul.create(llm, tools, system_prompt, max_turns)` — standard factory
- `MCPServer` is what every pipeline's `mcp/server.py` wraps

## Adding a New LLM Provider
1. Write tests in `tests/agentsoul/test_providers.py` first
2. Inherit `BaseLLM` from `agentsoul.providers.base`
3. Implement `async generate()` and `async stream()`
4. Register in `app/generation/registry.py` — NOT here

## Adding a New Memory Backend
1. Inherit `BaseMemory` from `agentsoul.memory.base`
2. Implement `async add()`, `async retrieve()`, `async clear()`
3. Wire up in `app/core/agent_factory.py` — NOT here

## Adding a New Retrieval Strategy
1. Inherit from `agentsoul.retrieval.base`
2. Implement `async search(query, k)` → `List[Document]`
3. Register in `FusionRetriever` if combining strategies

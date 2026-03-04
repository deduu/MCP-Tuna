# App — FastAPI Backend Rules

## This Directory
User-facing API layer. Creates AgentSoul instances, wires them to MCP servers,
and exposes OpenAI-compatible endpoints. Zero pipeline-specific logic here.

## Structure
```
app/
├── main.py              # Entry point — middleware, lifespan, router registration
├── api/                 # Route handlers (thin — delegate to orchestrators)
│   ├── api_chat.py      # POST /v1/chat/completions (streaming + non-streaming)
│   └── api_deck.py      # POST /generate-deck
├── core/
│   ├── config.py        # AppSettings, DatabaseSettings, MCPSettings, FileSettings
│   └── agent_factory.py # create_agent(model_name, mcp_servers, ...) → AgentSoul
├── db/                  # AsyncSession via SQLAlchemy + asyncpg
│   └── session.py       # DatabaseSessionManager (pool_size=5, max_overflow=10)
├── generation/          # LLM provider registry (gpt-4o, Qwen3)
└── services/            # MCP server wrappers: database, email, files, web
    └── base.py          # BaseService — generic async CRUD
└── utils/api/           # Orchestrators, handler chain, response builders
```

## Orchestrator Pattern
Two orchestrators compose single-responsibility handlers:

**`ChatAPIOrchestrator`:**
RequestParser → ModelRouteDecider → ToolSelector → ModelSelector → AgentExecutor → ResponseBuilder

**`DeckApiGenerator`:**
InputParser → DeckPromptBuilder → AgentExecutor → DeckResponseBuilder

Rules: No business logic in route handlers. Orchestrators compose; handlers are single-responsibility.

## Agent Creation
`core/agent_factory.py` → `create_agent(model_name, mcp_servers, system_prompt, max_turns=10)`
- Gets LLM from `generation/registry.py`
- Wires MCP servers from `MCPSettings` (auto_connect=True by default)
- Returns configured `AgentSoul` from `agentsoul.core.agent`

## Config (`core/config.py`)
All settings via dataclasses loaded from `.env`:
- `DatabaseSettings`: PostgreSQL (`insights_pg`, port 55432 via Docker Compose)
- `MCPSettings.servers`: mcp-tuna-gateway (8002), email (8003), file (8004), web (8005)
- `AppSettings.frontend_url`: `http://localhost:5173` (CORS allowlist)

## Services (`services/`)
Each service type has: `service.py` (business logic) + `mcp_server.py` (MCP tool definitions).
`BaseService` provides generic async CRUD for any SQLAlchemy model.
Services in this layer are for infrastructure (DB, email, files, web) — NOT pipeline operations.

## Rules
- Middleware: `RestrictRootAccessMiddleware` + `CORSMiddleware` — never remove
- Route handlers: zero business logic, only delegate to orchestrators
- Streaming: use SSE via `StreamingResponse`; non-streaming via `JSONResponse`
- Auth: via FastAPI dependencies — never inline

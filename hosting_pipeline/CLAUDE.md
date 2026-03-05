# Hosting Pipeline Rules

## This Directory
Deploys fine-tuned models as MCP tools or FastAPI endpoints. Manages deployment lifecycle
and provides a chat session interface for deployed models.

## Structure
```
hosting_pipeline/
├── services/
│   ├── hosting_service.py     # HostingService — deployment lifecycle management
│   └── chat_service.py        # ChatSession — conversation interface for deployed models
├── mcp/
│   └── server.py              # HostingMCPServer — MCP tool definitions
└── scripts/
```

## Service API (`services/hosting_service.py`)
`HostingService` manages an internal `_deployments` dict tracking active deployments.

All methods are async:
- `deploy_as_mcp(config)` — deploys model as MCP server (HTTP or stdio transport)
- `deploy_as_api(config)` — deploys model as FastAPI endpoint with `/generate` and `/health` routes
- `list_deployments()` → active deployment metadata
- `stop_deployment(deployment_id)` — shuts down a running deployment
- `health_check(deployment_id)` — checks if deployment is responsive

## Chat Service (`services/chat_service.py`)
`ChatSession` provides conversation interface for deployed models:
- Two modes: **API mode** (HTTP to deployed endpoint) or **Direct mode** (local HuggingFaceProvider)
- Supports streaming via `async for` generator pattern
- Manages chat history and system prompt

## Config (`shared/config.py`)
```python
HostingConfig(
    model_path="...",
    adapter_path=None,
    host="0.0.0.0",
    port=8001,
    transport="http",  # http | stdio
)

ChatConfig(
    endpoint=None,        # API mode: URL of deployed model
    model_path=None,      # Direct mode: local model path
    max_new_tokens=512,
    temperature=0.7,
    streaming=True,
)
```

## MCP Tools Exposed
`host.deploy_mcp`, `host.deploy_api`, `host.list_deployments`, `host.stop`

## Rules
- Uses `GPULock` from `shared/` for resource management during model loading
- Deployments run as background tasks via `asyncio.create_task()` — never block the request thread
- Always clean up GPU memory when stopping deployments
- Imports from `shared/` only — never import other pipelines

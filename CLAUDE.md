# Transcendence — Development Rules

## Project Overview
Transcendence is an end-to-end LLM fine-tuning platform. It supports two workflow paths:
- **Document Path:** Load docs → Generate (SFT/DPO/GRPO/KTO) → Clean → Normalize → Evaluate → Filter → Train → Host
- **Orchestration Path:** Generate problems → Collect agent trajectories → Score (accuracy + cost + latency) → Format → Train → Host

Exposed as 84 MCP tools via unified gateway. FastAPI backend provides an OpenAI-compatible API powered by the bundled AgentSoul framework.

## Tech Stack
- Python 3.11, FastAPI, Pydantic v2, asyncio
- AgentSoul (bundled at `src/agentsoul/`) — agent creation, MCP tools, memory, retrieval, multi-agent
- PyTorch, HuggingFace (transformers, datasets, PEFT/LoRA, trl, accelerate) — training
- sentence-transformers, spaCy, FAISS — evaluation metrics
- SQLAlchemy + asyncpg — PostgreSQL
- Auditi + Langfuse — observability and tracing
- MCP (Model Context Protocol) — tool integration
- Docker Compose — PostgreSQL
- Package manager: uv

## Commands
- `uv sync --all-extras` — install all dependencies
- `uv run pytest` — run all tests
- `uv run pytest -x -q` — stop on first failure, quiet
- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `python scripts/run_gateway.py` — start MCP gateway (port 8002)
- `docker compose up -d` — start PostgreSQL

## Architecture (MANDATORY)

### Three-tier structure
```
src/agentsoul/     ← Bundled agent framework (NEVER modify for app logic)
app/               ← FastAPI backend (user-facing API)
pipelines/         ← Stateless service modules (data processing + training)
```

### Key integration flow
User → `app/` (FastAPI) → `ChatAPIOrchestrator` → `AgentFactory.create_agent()` → `AgentSoul` → MCP Gateway (`mcp_gateway.py`) → Pipeline `*Service` classes

### Modules
- `src/agentsoul/` — Agent framework: core, strategies, providers, memory, retrieval, multi-agent, MCP server
- `app/` — FastAPI: `/v1/chat/completions`, `/generate-deck`, AgentFactory, DB, MCP wrappers
- `data_generator_pipeline/` — SFT/DPO/GRPO dataset generation from documents
- `data_evaluator_pipeline/` — Quality scoring: complexity, IFD, quality, similarity
- `data_cleaning_pipeline/` — Deduplication & schema validation
- `data_normalization_pipeline/` — Format conversion (SFT ↔ DPO ↔ GRPO ↔ KTO)
- `finetuning_pipeline/` — LoRA training + 4 sub-services (gpu, training, inference, discovery)
- `hosting_pipeline/` — Model deployment as MCP tool or FastAPI endpoint
- `orchestration/` — Schema-aware training data generation from agent trajectories
- `shared/` — Cross-pipeline: models, config, providers, registry (generic only)
- `mcp_gateway.py` — Unified MCP gateway (lazy-loads all pipeline services)

## Production Rules (MANDATORY)
1. **Pydantic v2 everywhere.** No raw dicts for structured data. `dataclasses` only for simple internal DTOs.
2. **Async-first.** All I/O is `async/await`. No blocking calls.
3. **One canonical data model.** Use `BaseDataPoint` from `shared/models.py`. All pipelines use it.
4. **Consistent key naming.** `instruction`, `input`, `output` — not `prompt`/`response`/`query`.
5. **Structured logging.** Use `src/agentsoul/utils/logger.py`. No `print()` anywhere.
6. **Circuit breakers** for every LLM/external API call.
7. **Timeouts:** 30s LLM calls, 15s tools, 300s training jobs.
8. **Typed exceptions.** Define in `shared/exceptions.py`. Services raise; MCP servers catch.
9. **Observability.** Instrument with Auditi + Langfuse. Configure via `.env`.
10. **Diagnostic logging.** `shared/diagnostics.py` writes per-session JSONL to `logs/sessions/`.
    See `logs/CLAUDE.md` for the full AI agent guide on reading sessions and writing findings.

## Development Workflow (MANDATORY)
1. **TDD.** Write tests BEFORE implementation. No commented-out tests.
2. **Plan mode for multi-file changes.** 3+ files → plan first.
3. Reference `tests/orchestration/test_orchestration.py` for test style.

## Dependency Rules
- Pipelines → `shared/` only; never import from each other
- `app/` → `shared/` + `agentsoul.*`; connects to pipelines via MCP only
- MCP servers → their pipeline's `*Service` only
- `src/agentsoul/` → no application-specific imports ever

## Code Style
- Absolute imports: `shared.*`, `agentsoul.*`, or pipeline-qualified paths
- `from __future__ import annotations` at top of every file
- Type annotations on all public functions
- Pydantic v2 with `model_config`

## Meta
If this file exceeds 100 lines, split into per-directory CLAUDE.md files.
Per-directory CLAUDE.md files exist in: `src/agentsoul/`, `app/`, `shared/`,
`data_generator_pipeline/`, `data_evaluator_pipeline/`, `finetuning_pipeline/`,
`orchestration/`, `tests/`, `logs/`

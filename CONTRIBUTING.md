# Contributing to MCP Tuna

Thanks for your interest in contributing!

## Setup

```bash
# Clone and install
git clone https://github.com/deduu/MCP-Tuna.git
cd mcp-tuna
uv sync --all-extras

# Verify
uv run pytest -x -q
uv run ruff check .
```

## Development Workflow

1. **Create a branch** from `main`
2. **Write tests first** (TDD is mandatory — see `tests/CLAUDE.md`)
3. **Implement** the feature or fix
4. **Run checks:**
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run pytest -x -q
   ```
5. **Open a PR** against `main`

## Code Style

- `from __future__ import annotations` at the top of every file
- Pydantic v2 for all structured data (no raw dicts)
- Async-first: all I/O uses `async/await`
- Absolute imports: `from shared.config import ...`
- Type annotations on all public functions

## Architecture Rules

- Pipelines import from `shared/` only — never from each other
- `src/agentsoul/` is the bundled agent framework — never modify for app logic
- MCP servers delegate to their pipeline's `*Service` class only
- New MCP tools go in the relevant `mcp/server.py`, not in `mcp_gateway.py`

## Testing

- Mirror source files 1:1 in `tests/`
- Mock all external I/O (LLM APIs, GPU, filesystem, network)
- Reference style: `tests/orchestration/test_orchestration.py`
- Use `tmp_path` for file I/O tests

## Adding a New MCP Tool

1. Add the service method in the pipeline's `services/` directory
2. Register the tool in the pipeline's `mcp/server.py`
3. Register in the composite server (`servers/*.py`) if applicable
4. Register in `mcp_gateway.py` for the unified gateway
5. Add tests in `tests/`

## Reporting Issues

Open an issue at https://github.com/deduu/MCP-Tuna/issues with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

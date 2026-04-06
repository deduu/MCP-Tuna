# AGENTS.md

Repository guidance for human and automated contributors.

## Core goals

- Preserve existing behavior unless the task explicitly changes behavior.
- Prefer modular, capability-driven designs over hardcoded model- or tool-specific branches.
- Keep interfaces backward compatible when touching MCP tools, frontend routes, request payloads, or persisted file formats.

## Fast routing

- `app/`: FastAPI entrypoints, API orchestration, and infrastructure adapters. Keep route handlers thin.
- `frontend/`: React control plane. Backend access should flow through `frontend/src/api/`.
- `shared/`: Canonical cross-pipeline models, config, registries, and utilities. Keep it generic.
- `data_*_pipeline/`, `finetuning_pipeline/`, `hosting_pipeline/`, `model_evaluator_pipeline/`, and `orchestration/`: workflow-specific services with local `AGENTS.md` files.
- `src/agentsoul/`: Bundled framework. Do not add application-specific behavior here.
- `mcp_gateway.py`: Large gateway surface. Search for exact tool names or helper names before opening broad sections.

## Search discipline

- Start with `rg` on symbol names, tool names, route names, config fields, or test names before opening files.
- Prefer the nearest local `AGENTS.md` once you narrow into a directory.
- Avoid scanning high-token files unless the task clearly requires them: `mcp_gateway.py`, `README.md`, prompt template directories, generated `output/`, ignored `data/`, and `logs/`.
- When touching MCP behavior, inspect the owning pipeline service and `mcp/server.py` before editing gateway wiring.
- When asked to configure Codex, Claude, or Cursor for MCP Tuna, prefer split-server examples in `examples/` to keep tool count low. Use the unified gateway only when the task genuinely spans many tool families.

## Architecture

- Avoid growing coordinator files when a feature can be expressed as a focused helper, registry entry, or leaf component.
- Prefer pure helpers for capability detection, schema normalization, and argument shaping.
- New model or trainer support should go through registries, factories, or typed capability maps rather than scattered conditionals.
- Do not duplicate tool wiring or schema mapping logic across multiple entry points if a shared helper can own it.
- Pipelines should depend on `shared/`, not on each other, unless an existing same-process boundary already documents a narrow exception.

## Frontend

- Keep feature pages thin. Move branching logic into small components or pure config/helpers.
- Prefer capability-driven UI. If the backend can advertise a capability, tool, or schema, read it instead of hardcoding assumptions.
- Use conservative fallbacks so new capability scaffolding does not break current text-only flows.
- Do not expose controls for unsupported backend actions.

## Backend

- Keep service boundaries explicit and narrow.
- Use `shared/models.py` and `shared/config.py` as canonical schema boundaries when the schema is shared.
- Dataset and training code must not assume all tasks are text-only; add modality-aware abstractions at the schema boundary.
- When adding new trainers, preserve the current LLM path and gate new behavior behind explicit capability checks.

## Quality bar

- Minimize regressions: add or update targeted tests when behavior changes, and run the smallest verification command that meaningfully covers the change.
- Prefer small, typed interfaces over broad `Record<string, unknown>` plumbing where the schema is known.
- Keep logging and user-facing errors concise and actionable.
- Add comments only when they explain non-obvious decisions or constraints.

## Token discipline

- Generate compact code and concise prose.
- Avoid boilerplate wrappers, repeated branching, and decorative comments.
- Reuse existing utilities and types before introducing new ones.
- Optimize for the fewest moving parts that still keep the code readable and extensible.

## Expected verification

- Frontend changes: run `npm --prefix frontend run build`.
- Python service or MCP boundary changes: run targeted `uv run pytest tests/<area> -q`.
- Gateway or tool registration changes: prefer the smallest relevant slice such as `uv run pytest tests/test_new_gateway_tools.py -q`.

# AGENTS.md

Repository guidance for human and automated contributors.

## Core goals

- Preserve existing behavior unless the task explicitly changes behavior.
- Prefer modular, capability-driven designs over hardcoded model- or tool-specific branches.
- Keep interfaces backward compatible when touching MCP tools, frontend routes, request payloads, or persisted file formats.

## Architecture

- Avoid growing coordinator files when a feature can be expressed as a focused helper, registry entry, or leaf component.
- Prefer pure helpers for capability detection, schema normalization, and argument shaping.
- New model or trainer support should go through registries, factories, or typed capability maps rather than scattered conditionals.
- Do not duplicate tool wiring or schema mapping logic across multiple entry points if a shared helper can own it.

## Frontend

- Keep feature pages thin. Move branching logic into small components or pure config/helpers.
- Prefer capability-driven UI. If the backend can advertise a capability, tool, or schema, read it instead of hardcoding assumptions.
- Use conservative fallbacks so new capability scaffolding does not break current text-only flows.
- Do not expose controls for unsupported backend actions.

## Backend

- Keep service boundaries explicit and narrow.
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
- Python service or MCP boundary changes: run targeted `pytest` for the affected area.

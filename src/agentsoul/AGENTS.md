# AGENTS.md

Rules for the bundled `agentsoul` framework.

- Treat this directory as framework code, not application code. Do not add MCP Tuna-specific behavior here.
- Import it as `agentsoul.*`, not `src.agentsoul.*`.
- Use the framework logging utilities instead of `print()` or unrelated logging patterns.
- New providers, memory backends, or retrieval strategies should stay framework-generic; app wiring belongs in `app/`.
- When extending framework behavior, add or update targeted tests under `tests/agentsoul/`.

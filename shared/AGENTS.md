# AGENTS.md

Rules for the shared cross-pipeline layer.

- Keep this directory generic. If a helper only makes sense for one pipeline, keep it in that pipeline.
- Prefer existing shared types and helpers before introducing new abstractions.
- `models.py` and `config.py` are the canonical schema definitions for shared data and settings.
- Use registries and factories for extensibility instead of scattering pipeline-specific conditionals.
- Typed exceptions belong here when multiple layers need them. Services raise them; MCP layers convert them to tool-facing errors.
- Avoid pipeline-specific names, logging strings, or business rules in shared modules.
- Verification: run the narrowest relevant tests under `tests/shared/`.

# AGENTS.md

FastAPI and API orchestration rules for `app/`.

- This layer is user-facing API and infrastructure wiring. Do not add pipeline-specific business logic here.
- Keep route handlers thin. Put branching and composition in `utils/api/` orchestrators and focused handlers.
- `core/agent_factory.py` owns agent construction and MCP server wiring.
- `core/config.py` is the settings boundary. Do not scatter raw environment parsing.
- `services/` in this directory are infrastructure adapters such as database, email, files, and web. They are not pipeline implementations.
- Preserve middleware and streaming behavior in `main.py` and the existing response builders unless the task explicitly changes them.
- When changing chat behavior, inspect `utils/api/handlers/` and `utils/api/responses/` before editing the route module.
- Verification: run the smallest affected pytest under `tests/app/` plus any chat or gateway tests touched by the change.

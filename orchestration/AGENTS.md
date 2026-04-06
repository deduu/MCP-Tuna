# AGENTS.md

Rules for `orchestration/`.

- This directory is for schema-aware training data generation from agent trajectories.
- Optimize for learning routing patterns from tool schemas, not memorizing specific tool names.
- Keep generation, trajectory collection, scoring, formatting, and training handoff as separate responsibilities.
- Drive hyperparameters and reward weights from config instead of hardcoding new branches.
- When changing scoring or trajectory structure, update targeted tests under `tests/orchestration/`.
- Use `tests/orchestration/test_orchestration.py` as the style reference for new tests.

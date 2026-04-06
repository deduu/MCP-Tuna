# AGENTS.md

Rules for `data_evaluator_pipeline/`.

- This pipeline evaluates dataset quality before training. Do not mix it with post-training model evaluation concerns.
- Keep metrics normalized to the expected score range and compose weighting in config or evaluator code, not ad hoc callers.
- LLM-backed synchronous metrics should use the shared sync adapter instead of rolling custom event-loop bridges.
- Filtering should continue to flow through the selection layer rather than bypassing it in service code.
- Mock all external providers and embeddings in tests.
- Verification: run the narrowest relevant pytest under `tests/data_evaluator_pipeline/`.

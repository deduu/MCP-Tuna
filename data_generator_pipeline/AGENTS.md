# AGENTS.md

Rules for `data_generator_pipeline/`.

- Keep the service API async and MCP-facing. Technique-specific behavior belongs in generators, not the transport layer.
- Generators should use the `BaseLLM` interface instead of calling providers directly.
- Prompt text belongs in `prompts/` and template helpers, not inline in generator logic.
- New techniques should go through the registry, a dedicated generator module, prompt templates, and targeted tests.
- Preserve source tracking fields on generated datapoints.
- Verification: run the smallest relevant pytest under `data_generator_pipeline/test/` or `tests/data_generator_pipeline/`.

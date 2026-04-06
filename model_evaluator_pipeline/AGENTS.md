# AGENTS.md

Rules for `model_evaluator_pipeline/`.

- This pipeline evaluates generated model outputs after training. Keep it distinct from dataset-quality evaluation in `data_evaluator_pipeline/`.
- Metrics should remain predictable and degrade safely on parse failures or missing judge output.
- Keep same-process integration with finetuning inference narrow and explicit.
- Mock external scoring libraries, judge LLM calls, and OpenAI clients in tests.
- Preserve export and summary contracts when adjusting result shapes.
- Verification: run the narrowest relevant pytest under `tests/model_evaluator_pipeline/`.

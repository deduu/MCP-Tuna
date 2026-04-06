# AGENTS.md

Rules for `data_cleaning_pipeline/`.

- Keep cleaning steps composable and independently callable.
- Do not change data semantics here. This layer removes or filters records; it does not rewrite content.
- Preserve auditability fields such as original and cleaned counts in returned results.
- Keep imports limited to `shared/` and standard library dependencies.
- Verification: run the smallest relevant pytest under `tests/data_cleaning_pipeline/`.

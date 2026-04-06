# AGENTS.md

Rules for `data_normalization_pipeline/`.

- Use canonical field names from `shared/models.py` and existing config schemas.
- Keep normalization focused on format conversion, key standardization, and text cleanup.
- Use NFC Unicode normalization for text cleanup unless the task explicitly changes that contract.
- Do not invent new schema keys when the target formats already exist.
- Verification: run the smallest relevant pytest under `tests/data_normalization_pipeline/`.

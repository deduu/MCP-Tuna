# AGENTS.md

Rules for `finetuning_pipeline/`.

- `services/pipeline_service.py` is the public facade. Keep sub-services focused and treat them as internal implementation details.
- Keep heavy GPU work out of request threads. Training and long-running jobs should stay in background tasks or isolated execution paths.
- Reuse the existing GPU cleanup path instead of adding ad hoc memory management.
- Training, inference, model discovery, and resource checks should stay in their owning service modules.
- Prefer `shared/config.py` and existing training helpers before adding new knobs or duplicated defaults.
- Tests must mock torch, GPU state, and external model downloads.
- Verification: run the narrowest relevant pytest under `tests/finetuning_pipeline/`.

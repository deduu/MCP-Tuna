# AGENTS.md

Rules for `hosting_pipeline/`.

- Keep deployment lifecycle management in the hosting service and chat behavior in the chat service.
- Deployments should stay asynchronous and non-blocking for request handlers.
- Reuse `GPULock` and existing cleanup hooks for model load and shutdown paths.
- Preserve the distinction between API mode and direct local chat mode.
- Verification: run the narrowest relevant pytest under `tests/hosting_pipeline/`.

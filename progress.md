# Progress Log

## 2026-03-06

### Significant Change 1: MCP Hook Alignment
- Updated dataset stats hook to use `dataset.info` with `file_path` instead of removed `dataset.get_stats`.
- Updated deployment hooks to use `host.health` and removed dependency on non-existent `host.get_status`, `host.get_logs`, and `host.undeploy`.
- Updated pipeline hooks to stop calling removed job-management tools (`workflow.list_jobs`, `workflow.get_status`, `workflow.cancel`) and return safe local placeholders.
- Updated judge hooks to stop calling removed tools (`judge.get_config`, `judge.list_criteria`) and align with `judge.list_types`.
- Bottleneck noted: frontend feature pages were built against an older MCP contract and require both tool-name and argument-shape migration.

### Significant Change 2: Dataset/Deployment/Pipeline Contract Migration
- Migrated dataset processing UI to load `data_points` first, then call clean/normalize/evaluate tools with correct payload shape.
- Fixed generation tool argument contracts (`file_path`, `file_paths`, `page_text/page_index/file_name`, required `technique`).
- Reworked split/merge dialogs to use current `dataset.split` and `dataset.merge` parameters.
- Updated deployment detail and inference test UI to `host.health` and `test.inference`.
- Converted custom pipeline builder to emit valid `workflow.run_pipeline` JSON step payloads.
- Bottleneck noted: several screens assumed path-based operations while backend now expects in-memory lists for many tools.

### Significant Change 3: Evaluation + Orchestration Realignment
- Replaced stale judge/eval tool calls with current `judge.evaluate`, `judge.compare_pair`, `judge.evaluate_batch`, `judge.create_rubric`, `ft_eval.batch/summary/export`, and `evaluate_model.batch/export`.
- Rebuilt orchestration step flow to current tools: `generate_problems -> collect_trajectories -> build_training_data -> train_orchestrator`.
- Added remediation matrix at `frontend-mcp-remediation-matrix.md` with old->new mappings and backend-limited gaps.
- Bottleneck noted: backend removed several convenience APIs (config/list/status/log/cancel), requiring UI fallbacks or capability shifts.

### Significant Change 4: Verification Build Fix
- Ran `npm --prefix frontend run build` to validate TypeScript after contract migration.
- Fixed a blocking TS error by removing an unused `useCallback` import in `frontend/src/components/training/NewTrainingPanel.tsx`.
